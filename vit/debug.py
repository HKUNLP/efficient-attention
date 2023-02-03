# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import json
import gc
import os
import models
from fvcore.nn.flop_count import flop_count
from timm.models import create_model
from torch.profiler import profile, record_function, ProfilerActivity
import efficient_attention

def get_args_parser():
    parser = argparse.ArgumentParser('Vision Transformer training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=4, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--attn-name', default='softmax', type=str, metavar='ATTN',
                        help='Name of attention model to use')
    
    parser.add_argument('--pretrained', action='store_true')

    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--attn-drop-rate', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-rate', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path-rate', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')
    parser.add_argument('--no-pos-emb', action='store_true')
    parser.set_defaults(no_pos_emb=False)
    
    # program-level parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    

    # FIXME a hacky workaround to add model-specific arguments;
    # there may be a better solution :)
    temp_args, _ = parser.parse_known_args()
    cls = create_model(temp_args.model)
    if hasattr(cls, 'add_model_specific_args'):
        cls.add_model_specific_args(parser)
        # FIXME: remove this condition when re-factoring is finished.
        efficient_attention.AttentionFactory.add_attn_specific_args(parser, temp_args.attn_name)
    del cls, temp_args

    return parser

def main(args):
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    args.num_classes = 1000

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        args=args,
    )
    model.to(device)
    print(model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    inputs = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=args.device)
    # Set model to evaluation mode for analysis.
    # Evaluation mode can avoid getting stuck with sync batchnorm.
    # model_mode = model.training
    # model.eval()
    count_dict, *_ = flop_count(model, inputs)
    print("Flops: {:,} G".format(sum(count_dict.values())))
    # warmup
    # model.train(model_mode)
    if args.device != 'cpu':
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, 
                with_stack=True, profile_memory=True) as prof:
            out = model(inputs)
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1))
    else:
        out = model(inputs)
    print(out.shape, out.sum(), torch.isnan(out).all())
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConViT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args(namespace=efficient_attention.NestedNamespace())
    # for backwards compatibility.
    if not hasattr(args, 'attn_args'):
        args.attn_args = argparse.Namespace()
    args.attn_specific_args = args.attn_args
    main(args)

