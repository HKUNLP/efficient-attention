# Efficient Attentions

This repository contains the official implementation of experiments conducted in
- [EVA: Efficient attention via control variates (ICLR 2023)](https://openreview.net/forum?id=G-uNfHKrj46)
- [LARA: Linear complexity randomized self-attention mechanism (ICML 2022)](https://arxiv.org/abs/2204.04667)

🌲 Repo structure:
- `efficient-attention`: a small self-contained codebase that implements various efficient attention mechanisms. Please see the [usage](#basic-usage-of-the-efficient-attention-library) for more details.
- `vit`: codebase for **image classification** experiments, which is adapted from 
  - [DeiT](https://github.com/facebookresearch/deit) for the file structure, and 
  - [PvT](https://github.com/whai362/PVT) for PvTv2 model classes.
- `fairseq`: a modified fork of [fairseq](https://fairseq.readthedocs.io/en/latest/) for language tasks, including **machine translation** and **autoregressive language modeling**.
- `main.sh`: a bash script for launching all experiments.
  - See the script for the list of arguments.
  - Note that arguments after `-e True` are directly passed to the training command. You can pass custom arguments to the training command by appending them after `-e True`. 

## Dependencies
To setup the environment, run the following commands to install the required dependencies (recommended in a virtual environment):

```bash
# install packages
pip install -r requirements.txt
# install efficient-attention library
pip install -e efficient-attention

# OPTIONAL: install fairseq library for running language tasks
cd fairseq
python3 setup.py build develop --user
cd ..
```

## Basic Usage of the Efficient Attention Library
`efficient-attention` is a small self-contained codebase that collects several efficient attention mechanisms.

### Passing Attention-specific Arguments to Argparse
- For arguments specific to each attention mechanism, please check the `add_attn_specific_args()` class method in the corresponding python file.
- To pass these arguments to the `argparse` parser, follow the following code snippet:
```python
import argparse
from efficient_attention import AttentionFactory
# ...
parser = argparse.ArgumentParser()
parser.add_argument('--attn-name', default='softmax', type=str, metavar='ATTN',
                        help='Name of attention model to use')
# ...
temp_args, _ = parser.parse_known_args()
# add attention-specific arguments to the parser
# struct_name: name of the inner namespace to store all attention-specific arguments
# prefix: prefix to prepend to all argument names
#         for example, if prefix = encoder-attn, then for the argument --window-size 
#         we need to pass --encoder-attn-window-size
#         this is useful to avoid argument name conflicts.
efficient_attention.AttentionFactory.add_attn_specific_args(parser, temp_args.attn_name, struct_name="attn_args", prefix="")
# parse arguments to a namespace that supports nested attributes
args = parser.parse_args(namespace=efficient_attention.NestedNamespace())
# now we can access the attention-specific arguments via args.attn_args
print(args.attn_args.window_size)
```

### Create an Efficient Attention Module
In a `torch.nn.Module` class, you can create an efficient attention module as follows:
```python
# we might want to pass attention-specific arguments to the attention module
# along with other related arguments
attn_args = {
    **vars(args.attn_args),
    **{
    'dim': args.embed_dim, 
    'num_heads': args.num_heads, 
    'qkv_bias': args.qkv_bias, 
    'attn_drop': args.attn_drop_rate, 
    'proj_drop': args.drop_rate,
    }
}
self.attn = AttentionFactory.build_attention(attn_name = attn_name, attn_args = attn_args)

# the module can then be used as a normal function as
x = self.attn(x)
```

## Image Classification on ImageNet1k

### Data Preparation
We follow the setup similar to [DeiT](https://github.com/facebookresearch/deit) to pre-process the [ImageNet](http://image-net.org/) dataset. Download ImageNet train and val images and place them in the following directory structure so that it can be compatible with the torchvision [`datasets.ImageFolder`](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)
```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

### Training & Evaluation
The following commands are used for training and evaluating various vision transformers with `LARA/EVA`. The training is assumed to be conducted with 8 GPUs.

#### ImageNet Classification on DeiT (sequence length 784(suffix:\*_p8)/196(suffix:\*_p16))
To use `LARA/EVA` in different DeiT architectures:
```bash
# LARA: DeiT-tiny-p8
bash main.sh -m evit_tiny_p8 -p <dir-of-imagenet-data> -g 8 -d imagenet -e TRUE --dist-eval --num-workers 16 --clip-grad 5.0 --warmup-epochs 10 --seed 1 --attn-name lara --mis-type mis-opt --proposal-gen pool-mixed --alpha-coeff 2.0 --num-landmarks 49

# LARA: DeiT-tiny-p16
bash main.sh -m evit_tiny_p16 -p <dir-of-imagenet-data> -g 8 -d imagenet -e TRUE --dist-eval --num-workers 16 --clip-grad 5.0 --warmup-epochs 10 --seed 1 --attn-name lara --mis-type mis-opt --proposal-gen pool-mixed --alpha-coeff 2.0 --num-landmarks 49

# LARA: DeiT-small-p16
bash main.sh -m evit_small_p16 -p <dir-of-imagenet-data> -g 8 -d imagenet -e TRUE --dist-eval --num-workers 16 --clip-grad 5.0 --warmup-epochs 10 --seed 1 --attn-name lara --mis-type mis-opt --proposal-gen pool-mixed --alpha-coeff 2.0 --num-landmarks 49

# EVA: DeiT-tiny-p8
bash main.sh -m evit_tiny_p8 -p <dir-of-imagenet-data> -g 8 -d imagenet -e TRUE --dist-eval --num-workers 16 --clip-grad 5.0 --warmup-epochs 10 --seed 1 --attn-name eva --num-landmarks 49 --adaptive-proj default --window-size 7 --attn-2d --use-rpe

# EVA: DeiT-tiny-p16
bash main.sh -m evit_tiny_p16 -p <dir-of-imagenet-data> -g 8 -d imagenet -e TRUE --dist-eval --num-workers 16 --clip-grad 5.0 --warmup-epochs 10 --seed 1 --attn-name eva --num-landmarks 49 --adaptive-proj default --window-size 7 --attn-2d --use-rpe

# EVA: DeiT-small-p16
bash main.sh -m evit_small_p16 -p <dir-of-imagenet-data> -g 8 -d imagenet -e TRUE --dist-eval --num-workers 16 --clip-grad 5.0 --warmup-epochs 10 --seed 1 --attn-name eva --num-landmarks 49 --adaptive-proj default --window-size 7 --attn-2d --use-rpe
```

#### ImageNet Classification on PVTv2-B3 (sequence length: 3136 -> 784 -> 196 -> 49)
To adapt `LARA/EVA` in PvTv2 architectures:
```bash
# LARA Attention
bash main.sh -m pvt_medium2 -p <dir-of-imagenet-data> -g 8 -d imagenet -e TRUE --dist-eval --num-workers 16 --clip-grad 1.0 --drop-path-rate 0.3 --warmup-epochs 10 --seed 1 --attn-name lara --pool-module-type dense --mis-type mis-opt --proposal-gen pool-mixed --num-landmarks 49 --alpha-coeff 2.0 --repeated-aug

# EVA Attention
bash main.sh -m pvt_medium2 -p <dir-of-imagenet-data> -g 8 -d imagenet -e TRUE --dist-eval --num-workers 16 --clip-grad 5.0 --drop-path-rate 0.3 --warmup-epochs 10 --seed 1 --attn-name eva --num-landmarks 49 --adaptive-proj default --window-size 7 --attn-2d --use-rpe --repeated-aug
```

#### The usage of other attention mechanisms:
Alternatively, you may want to try out other attention mechanisms:
```bash
# Softmax Attention
bash main.sh -m evit_tiny_p8 -d imagenet -e TRUE --dist-eval --num-workers 16 --clip-grad 5.0 --warmup-epochs 10 --seed 1 --attn-name softmax
# RFA/Performer
bash main.sh -m evit_tiny_p8 -d imagenet -e TRUE --dist-eval --num-workers 16 --clip-grad 5.0 --warmup-epochs 10 --seed 1 --attn-name performer --proj-method favorp --approx-attn-dim 64
# Local Attention
bash main.sh -m evit_tiny_p8 -d imagenet -e TRUE --dist-eval --num-workers 16 --clip-grad 5.0 --warmup-epochs 10 --seed 1 --attn-name local --window-size 7 --attn-2d --use-rpe
```

## Language Tasks

### Data Preparation
We use the standard pre-processing [fairseq](https://fairseq.readthedocs.io/en/latest/) to prepare the data for language tasks.

- For machine translation, please follow [here](https://github.com/pytorch/fairseq/blob/master/examples/scaling_nmt/README.md#training-a-new-model-on-wmt16-en-de) to prepare for the binarized `WMT'14 EN-DE` data;
- For autoregressive language modeling, follow [here](https://github.com/pytorch/fairseq/blob/master/examples/language_model/README.md#1-preprocess-the-data) to process the `Wikitext-103` dataset.

### Training

- `-r <resume-ckpt-DIR>` specifies the DIRECTORY that stores your checkpoints during training and can be used to resume training.
- Note that all attention-specific arguments need to be associated with prefix `--encoder-attn-` (for encoder-side) / `--decoder-attn-` (for decoder-side). See the examples below.

#### Machine Translation

```bash
## LARA
CUDA_VISIBLE_DEVICES=0,1,2,3 bash main.sh -p <dir-of-your-bin-data> -d wmt -s lara_8 -g 4 -e TRUE --attn-name-encoder lara --encoder-attn-num-landmarks 8 --encoder-attn-proposal-gen adaptive-1d --encoder-attn-mis-type mis-opt

CUDA_VISIBLE_DEVICES=0,1,2,3 bash main.sh -p <dir-of-your-bin-data> -d wmt -s lara_16 -g 4 -e TRUE --attn-name-encoder lara --encoder-attn-num-landmarks 16 --encoder-attn-proposal-gen adaptive-1d --encoder-attn-mis-type mis-opt

CUDA_VISIBLE_DEVICES=0,1,2,3 bash main.sh -p <dir-of-your-bin-data> -d wmt -s lara_32 -g 4 -e TRUE --attn-name-encoder lara --encoder-attn-num-landmarks 32 --encoder-attn-proposal-gen adaptive-1d --encoder-attn-mis-type mis-opt

## EVA
CUDA_VISIBLE_DEVICES=0,1,2,3 bash main.sh -p <dir-of-your-bin-data> -d wmt -s eva_8_8 -g 4 -e TRUE --attn-name-encoder eva --encoder-attn-window-size 8 --encoder-attn-num-landmarks 8 --encoder-attn-adaptive-proj no-ln --encoder-attn-use-t5-rpe --encoder-attn-overlap-window

CUDA_VISIBLE_DEVICES=0,1,2,3 bash main.sh -p <dir-of-your-bin-data> -d wmt -s eva_16_8 -g 4 -e TRUE --attn-name-encoder eva --encoder-attn-window-size 16 --encoder-attn-num-landmarks 8 --encoder-attn-adaptive-proj no-ln --encoder-attn-use-t5-rpe --encoder-attn-overlap-window

CUDA_VISIBLE_DEVICES=0,1,2,3 bash main.sh -p <dir-of-your-bin-data> -d wmt -s eva_32_8 -g 4 -e TRUE --attn-name-encoder eva --encoder-attn-window-size 32 --encoder-attn-num-landmarks 8 --encoder-attn-adaptive-proj no-ln --encoder-attn-use-t5-rpe --encoder-attn-overlap-window
```

#### Autoregressive Language Modeling

```bash
# Currently, LARA does not support causal masking yet.

# EVA on a 16-layer Transformer LM
CUDA_VISIBLE_DEVICES=0,1,2,3 bash main.sh -p <dir-of-your-bin-data> -m 16layers -d wikitext103 -s eva_128_8_16layers -g 4 -e TRUE --attn-name-decoder causal_eva --decoder-attn-window-size 128 --decoder-attn-causal --decoder-attn-adaptive-proj qk --decoder-attn-chunk-size 8 --decoder-attn-use-t5-rpe

# EVA on a 32-layer Transformer LM
CUDA_VISIBLE_DEVICES=0,1,2,3 bash main.sh -p <dir-of-your-bin-data> -m 32layers -d wikitext103 -s eva_128_8_32layers -g 4 -e TRUE --attn-name-decoder causal_eva --decoder-attn-window-size 128 --decoder-attn-causal --decoder-attn-adaptive-proj qk --decoder-attn-chunk-size 8 --decoder-attn-use-t5-rpe
```

### Generation & Evaluation

For generation & evaluation, simply pass argument `-i true` when calling `main.sh` to perform the inference procedure only. The checkpoint path can be specified as `-c <your-ckpt-path>`. For example,

```bash
# Machine Translation
CUDA_VISIBLE_DEVICES=0 bash main.sh -i true -c <your-possibly-avg-checkpoint.pt> -p <dir-of-your-bin-data> -d wmt -g 1

# Autoregressive Language Modeling
CUDA_VISIBLE_DEVICES=0 bash main.sh -i true -c <your-checkpoint_last.pt> -p <dir-of-your-bin-data> -d wikitext103 -g 1
```

### Pre-trained Models

We also provide trained `EVA` model checkpoints in [OneDrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/EkHamShbSkdBkpnxg_1TbbQBQOHnpe8gnS0ttOqZXjdBxA?e=plb7NR) for machine translation and language modeling tasks: 

- [wikitext103-eva-16layers-lm](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/linzheng_connect_hku_hk/EV2rzXhU8qVGvpjST3pxqbYBcSya_IdVCw7g42y4Jwwymg?e=RUof1O)
- [wikitext103-eva-32layers-lm](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/linzheng_connect_hku_hk/EUZG-jzW3hpChCUPSnFMm78Bk2I3phRfA62328NNgehQsg?e=IJN30S)
- [wmt14ende-eva-e32_c8-mt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/linzheng_connect_hku_hk/EVVzu135gxBMkDVJRlcS-QYBYfNPZxGTHdT31lk9y7tuNw?e=5Ifdlz)
- [wmt14ende-eva-e8_c8-mt](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/linzheng_connect_hku_hk/Ef-MR-LUkIZJpI4qQwWt6lsB0fynBQ3VRy05b3yFOk5vbA?e=O9TbI4)

## Citation
```bibtex
@inproceedings{zheng2023efficient,
  title={Efficient Attention via Control Variates},
  author={Lin Zheng and Jianbo Yuan and Chong Wang and Lingpeng Kong},
  booktitle={International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=G-uNfHKrj46}
}
```

```bibtex
@inproceedings{zheng2022linear,
  title={Linear complexity randomized self-attention mechanism},
  author={Lin Zheng and Chong Wang and Lingpeng Kong},
  booktitle={International Conference on Machine Learning},
  pages={27011--27041},
  year={2022},
  organization={PMLR}
}
```

