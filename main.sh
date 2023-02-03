#!/bin/bash -eu
set -x 

# process named arguments
err_msg() { echo "Invalid arguments" 1>&2; exit 1; }

while getopts ":m:s:c:g:r:i:p:d:e:" o; do
    case "${o}" in
        m)
            MODEL=${OPTARG}
            ;;
        s)
            SUFFIX=${OPTARG}
            ;;
        d)
            DATASET=${OPTARG}
            ;;
        r)
            RESUME_CKPT_DIR=${OPTARG}
            ;;
        g)
            NUM_GPUS=${OPTARG}
            ;;
        p)
            DATA_PATH=${OPTARG}
            ;;
        c)
            CKPT_PATH=${OPTARG}
            ;;
        i)
            INFERENCE_ONLY=${OPTARG}
            ;;
        e)
            # use -e to separate the program-level arguments and custom arguments.
            break
            ;;
        *)
            err_msg
            ;;
    esac
done
shift $((OPTIND-1))
DATASET=${DATASET:-"wikitext103"}
SUFFIX=${SUFFIX:-'default'}
NUM_GPUS=${NUM_GPUS:-4}
DATA_PATH=${DATA_PATH:-'./datasets/imagenet'}
INFERENCE_ONLY=${INFERENCE_ONLY:-false}
RESUME_CKPT_DIR=${RESUME_CKPT_DIR:-none}
CKPT_PATH=${CKPT_PATH:-none}

case $DATASET in
  wikitext103)
    cd fairseq
    DATA=$DATA_PATH/wikitext-103
    if ! "$INFERENCE_ONLY"; then
        EX_POSTFIX=$(date +%T | sed "s/:/-/g" | echo "-$(cat -)")
        ARCH="transformer_lm_wiki103"
        MODEL=${MODEL:-16layers}
        if [ "$MODEL" == "16layers" ]; then
            MODEL_ARGS="--decoder-layers 16"
        elif [ "$MODEL" == "32layers" ]; then
            MODEL_ARGS="--decoder-layers 32 --decoder-layerdrop 0.2"
        else
            echo "Invalid model name; must be either -m 16layers or -m 32layers"
            exit 1;
        fi

        if [[ "$RESUME_CKPT_DIR" != "none" ]]; then
            CKPT_DIR=$RESUME_CKPT_DIR
        else
            CKPT_DIR="checkpoints/""${ARCH//_/-}""$EX_POSTFIX"'-'$SUFFIX
        fi
        mkdir -p $CKPT_DIR
        UPDATE_FREQ=$(( 8 / ${NUM_GPUS} ))
        python3 -m fairseq_cli.train --task language_modeling $DATA \
            --arch $ARCH --no-progress-bar --log-interval 50 \
            --max-update 286000 --lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
            --warmup-updates 16000 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --optimizer nag --min-lr 0.0001 --clip-norm 0.1 \
            --criterion adaptive_loss --max-tokens 9216 --update-freq $UPDATE_FREQ --tokens-per-sample 512 --seed 1 \
            --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d \
            --save-dir $CKPT_DIR --save-interval 2 --keep-last-epochs 2 $@ $MODEL_ARGS
        CKPT_PATH=$CKPT_DIR/checkpoint_last.pt
    fi
    echo ">>>>>>>> Testing PPL w/ context window size 0" 
    python3 -m fairseq_cli.eval_lm $DATA \
        --path $CKPT_PATH \
        --batch-size 16 --gen-subset test\
        --tokens-per-sample 512 \
        --context-window 0

    echo ">>>>>>>> Validation PPL w/ context window size 0" 
    python3 -m fairseq_cli.eval_lm $DATA \
        --path $CKPT_PATH \
        --batch-size 16 --gen-subset valid\
        --tokens-per-sample 512 \
        --context-window 0

    echo ">>>>>>>> Testing PPL w/ context window size 256" 
    python3 -m fairseq_cli.eval_lm $DATA \
        --path $CKPT_PATH \
        --batch-size 16 --gen-subset test\
        --tokens-per-sample 512 \
        --context-window 256

    echo ">>>>>>>> Validation PPL w/ context window size 256" 
    python3 -m fairseq_cli.eval_lm $DATA \
        --path $CKPT_PATH \
        --batch-size 16 --gen-subset valid\
        --tokens-per-sample 512 \
        --context-window 256

    echo ">>>>>>>> Testing PPL w/ context window size 480" 
    python3 -m fairseq_cli.eval_lm $DATA \
        --path $CKPT_PATH \
        --batch-size 16 --gen-subset test\
        --tokens-per-sample 512 \
        --context-window 480

    echo ">>>>>>>> Validation PPL w/ context window size 480" 
    python3 -m fairseq_cli.eval_lm $DATA \
        --path $CKPT_PATH \
        --batch-size 16 --gen-subset valid\
        --tokens-per-sample 512 \
        --context-window 480
   ;;

  wmt)
    cd fairseq
    DATA_TAG="$DATA_PATH/wmt16-en-de"
    
    if ! "$INFERENCE_ONLY"; then
        EX_POSTFIX=$(date +%T | sed "s/:/-/g" | echo "-$(cat -)")
        MODEL=${MODEL:-default}
        if [ "$MODEL" != "default" ]; then
            echo "for WMT14 EN-DE experiments, only -m default (transformer_wmt_en_de) is supported."; 
            exit 1;
        fi
        ARCH="transformer_wmt_en_de"
        if [[ "$RESUME_CKPT_DIR" != "none" ]]; then
            CKPT_DIR=$RESUME_CKPT_DIR
        else
            CKPT_DIR="checkpoints/""${ARCH//_/-}"'_'"$EX_POSTFIX"'_'$SUFFIX
        fi
        mkdir -p $CKPT_DIR
        UPDATE_FREQ=$(( 8 / $NUM_GPUS ))
        python3 -m fairseq_cli.train $DATA_TAG \
                    --arch $ARCH --share-all-embeddings \
                    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 5.0 \
                    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 6000 \
                    --lr 0.0007 --stop-min-lr 1e-09 \
                    --dropout 0.1 --activation-dropout 0.1\
                    --tensorboard-logdir $CKPT_DIR --seed 2\
                    --best-checkpoint-metric ppl\
                    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0\
                    --max-tokens 4096 --save-dir $CKPT_DIR --amp\
                    --update-freq $UPDATE_FREQ --no-progress-bar --log-interval 200\
                    --save-interval 10  --keep-last-epochs 1 --max-update 300000\
                    --save-interval-updates 1000 --keep-interval-updates 20 $@
        if [ ! -f "$CKPT_DIR/checkpoint.avg10.pt" ]; then
        python3 scripts/average_checkpoints.py \
        --inputs $CKPT_DIR \
        --num-update-checkpoints 10 \
        --output $CKPT_DIR/checkpoint.avg10.pt
        fi
        CKPT_PATH=$CKPT_DIR/checkpoint.avg10.pt
    else
        # infer ckpt dir from the given path
        CKPT_DIR=${CKPT_PATH%/*}
    fi
    python3 -m fairseq_cli.generate $DATA_TAG \
    --path $CKPT_PATH \
    --beam 4 --lenpen 0.6 --max-len-a 1 --max-len-b 50 --remove-bpe > $CKPT_DIR/gen.out    

    # "compound split" tokenized BLEU
    echo "--------------> compound split BLEU <----------------"
    bash scripts/compound_split_bleu.sh $CKPT_DIR/gen.out
    ;;
  
  imagenet)
    cd vit
    MODEL=${MODEL:-evit_tiny_p16}
    CKPT_DIR=$DATASET"_"$MODEL"_"$SUFFIX
    torchrun --nproc_per_node $NUM_GPUS main.py --model $MODEL --data-set IMAGENET --batch-size 128 \
        --data-path $DATA_PATH --output-dir $CKPT_DIR "$@"
    # alternatively, specify the env with $GPU, $HOST_ADDR, $PORT, $NODE_RANK, $NUM_NODES, etc. as follows
    # torchrun \
    #   --nproc_per_node $GPU --master_addr $HOST_ADDR \
    #   --master_port $PORT --node_rank $NODE_RANK --nnodes $NUM_NODES main.py \
    #   --model $MODEL --data-set $DATASET --batch-size 128 \
    #   --data-path $DATA_PATH --output-dir $CKPT_DIR "$@"
    ;;
  
  *)
    echo "unknown dataset"
    exit 1
    ;;
esac
