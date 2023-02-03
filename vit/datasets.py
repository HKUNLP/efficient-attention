# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#

import io
import json
import multiprocessing as mp
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
from PIL import Image
from timm.data import Mixup, create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from constants import (CIFAR_DEFAULT_MEAN, CIFAR_DEFAULT_STD,
                       IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        num_classes = 10
    elif args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        num_classes = 100
    elif args.data_set == 'IMAGENET':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(args.data_path, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        num_classes = 1000
    else:
        raise NotImplementedError("We only support IMAGENET, CIFAR-10, CIFAR-100 Now.")
    return dataset, num_classes

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    #3
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    if args.data_set == 'IMAGENET':
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    elif args.data_set.startswith('CIFAR'):
        t.append(transforms.Normalize(CIFAR_DEFAULT_MEAN, CIFAR_DEFAULT_STD))
    return transforms.Compose(t)