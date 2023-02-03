# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.registry import register_model
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from efficient_attention import AttentionFactory
from .model_utils import GatedMlp

__all__ = [
    'evit_tiny_p16', 
    'evit_small_p16', 
    'evit_base_p16',
    'evit_tiny_p8', 
    'evit_small_p8', 
    'evit_base_p8',
    'evit_tiny_p4',
    'evit_small_p4',
    ]

def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val

def default(val, d):
    return val if val is not None else d

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding, from timm
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, patchify_stem='default'):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.new_H, self.new_W = self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]

        if patchify_stem == 'hmlp':
            # 224 x 224
            # Groupnorm with a single group is equivalent to LayerNorm
            if patch_size == 8:
                first_split_size = 2
            elif patch_size == 16:
                first_split_size = 4
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim//4, kernel_size=first_split_size, stride=first_split_size),
                nn.GroupNorm(1, embed_dim//4),
                nn.GELU(),
                nn.Conv2d(embed_dim//4, embed_dim//4, kernel_size=2, stride=2),
                nn.GroupNorm(1, embed_dim//4),
                nn.GELU(),
                nn.Conv2d(embed_dim//4, embed_dim, kernel_size=2, stride=2),
                nn.GroupNorm(1, embed_dim),
            )
        elif patchify_stem == 'conv':
            if patch_size == 8:
                last_conv_layer = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)
            elif patch_size == 16:
                last_conv_layer = nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2, padding=0)
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim//4, kernel_size=3, stride=2, padding=1), # 112 x 112
                nn.GroupNorm(1, embed_dim//4), 
                nn.ReLU(),
                nn.Conv2d(embed_dim//4, embed_dim//4, kernel_size=3, stride=2, padding=1), # 56 x 56
                nn.GroupNorm(1, embed_dim//4), 
                nn.ReLU(),
                nn.Conv2d(embed_dim//4, embed_dim, kernel_size=3, stride=2, padding=1), # 28 x 28
                nn.GroupNorm(1, embed_dim), 
                nn.ReLU(),
                last_conv_layer, # 28 x 28
                )
        elif patchify_stem == 'default':
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            raise NotImplementedError
        self.apply(self._init_weights)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).permute(0, 2, 3, 1)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Block(nn.Module):

    def __init__(self, 
                attn_name,
                attn_args,
                dim, 
                mlp_ratio, 
                drop_path, 
                drop_rate=0., 
                act_layer=nn.GELU, 
                norm_layer=nn.LayerNorm,
                use_glu=False,
                ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionFactory.build_attention(attn_name = attn_name, attn_args = attn_args)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = GatedMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_rate, use_glu=use_glu)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class EfficientTransformer(nn.Module):
    """ Vision Transformer

    """
    def __init__(self, args):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = args.num_classes
        self.num_features = self.embed_dim = args.embed_dim  # num_features for consistency with other models
        norm_layer = args.norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.depth = args.depth
        self.epoch = 0

        self.patch_embed = PatchEmbed(
            img_size=args.input_size, 
            patch_size=args.patch_size, 
            in_chans=args.in_chans, 
            embed_dim=args.embed_dim,
            patchify_stem=args.patchify_stem)

        # Classifier head
        self.head = nn.Linear(self.embed_dim, args.num_classes) if args.num_classes > 0 else nn.Identity()

        self.use_pos_emb = not args.no_pos_emb
        if self.use_pos_emb:
            # self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, args.embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.new_H, self.patch_embed.new_W, args.embed_dim))
            self.pos_drop = nn.Dropout(p=args.drop_rate)
            trunc_normal_(self.pos_embed, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.norm_before_pooling = norm_layer(args.embed_dim)

        # if self.pool_method == 'attn':
            # self.attention_pool = nn.Linear(args.embed_dim, 1)

        attn_args = {
            **vars(args.attn_specific_args),
            **{
            'dim': args.embed_dim, 
            'num_heads': args.num_heads, 
            'qkv_bias': args.qkv_bias, 
            'attn_drop': args.attn_drop_rate, 
            'proj_drop': args.drop_rate,
            }
        }
        dpr = [x.item() for x in torch.linspace(0, args.drop_path_rate, args.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                attn_name=args.attn_name, attn_args=attn_args,
                dim=args.embed_dim, mlp_ratio=args.mlp_ratio, drop_path=dpr[i],
                drop_rate=args.drop_rate, norm_layer=norm_layer, use_glu=args.use_glu)
            for i in range(args.depth)
            ])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)

        # cls_tokens = self.cls_token.expand(B, -1, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # print(cls_tokens.shape)
        # x = torch.cat((cls_tokens, x), dim=1)
        if self.use_pos_emb:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        B, H, W, C = x.shape

        for layer, blk in enumerate(self.blocks):
            x = blk(x)
        
        x = self.norm_before_pooling(x.reshape(B, H * W, -1))  # B H W C
        
        x = x.mean(1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Deit")
        # RFA parameters
        parser.add_argument('--patchify-stem', default='default', type=str)
        parser.add_argument('--num-heads', default=None, type=int)
        parser.add_argument('--use-glu', action='store_true', default=False)
        parser.add_argument('--patch-size', default=16, type=int)
        parser.add_argument('--depth', default=12, type=int, help='number of transformer layers')
        return parent_parser
        
        
def base_et(args):
    args.depth = getattr(args, 'depth', 12)
    args.mlp_ratio = getattr(args, 'mlp_ratio', 4)
    args.qkv_bias = getattr(args, 'qkv_bias', True)
    args.qk_scale = getattr(args, 'qk_scale', None)
    args.norm_layer = getattr(args, 'norm_layer', partial(nn.LayerNorm, eps=1e-6))
    args.in_chans = getattr(args, 'in_chans', 3)

@register_model
def evit_tiny_p16(args=None, **kwargs):
    if args is not None:
        args.embed_dim = 192
        args.num_heads = default(args.num_heads, 3)
        args.patch_size = 16
        base_et(args)
        model = EfficientTransformer(args)
        return model
    else:
        # Possibly running a prelim model creation. Only return cls
        return EfficientTransformer

@register_model
def evit_small_p16(args=None, **kwargs):
    if args is not None:
        args.embed_dim = 384
        args.num_heads = default(args.num_heads, 6)
        args.patch_size = 16
        base_et(args)
        model = EfficientTransformer(args)
        return model
    else:
        # Possibly running a prelim model creation. Only return cls
        return EfficientTransformer

@register_model
def evit_base_p16(args=None, **kwargs):
    if args is not None:
        args.embed_dim = 768
        args.num_heads = default(args.num_heads, 12)
        args.patch_size = 16
        base_et(args)
        model = EfficientTransformer(args)
        return model
    else:
        # Possibly running a prelim model creation. Only return cls
        return EfficientTransformer


@register_model
def evit_tiny_p8(args=None, **kwargs):
    if args is not None:
        args.embed_dim = 192
        args.patch_size = 8
        args.num_heads = default(args.num_heads, 3)
        base_et(args)
        model = EfficientTransformer(args)
        return model
    else:
        # Possibly running a prelim model creation. Only return cls
        return EfficientTransformer

@register_model
def evit_small_p8(args=None, **kwargs):
    if args is not None:
        args.embed_dim = 384
        args.patch_size = 8
        args.num_heads = default(args.num_heads, 6)
        base_et(args)
        model = EfficientTransformer(args)
        return model
    else:
        # Possibly running a prelim model creation. Only return cls
        return EfficientTransformer

@register_model
def evit_base_p8(args=None, **kwargs):
    if args is not None:
        args.embed_dim = 768
        args.patch_size = 8
        args.num_heads = default(args.num_heads, 12)
        base_et(args)
        model = EfficientTransformer(args)
        return model
    else:
        # Possibly running a prelim model creation. Only return cls
        return EfficientTransformer



@register_model
def evit_tiny_p4(args=None, **kwargs):
    if args is not None:
        args.embed_dim = 192
        args.patch_size = 4
        args.num_heads = default(args.num_heads, 3)
        base_et(args)
        model = EfficientTransformer(args)
        return model
    else:
        # Possibly running a prelim model creation. Only return cls
        return EfficientTransformer


@register_model
def evit_small_p4(args=None, **kwargs):
    if args is not None:
        args.embed_dim = 384
        args.patch_size = 4
        args.num_heads = default(args.num_heads, 6)
        base_et(args)
        model = EfficientTransformer(args)
        return model
    else:
        # Possibly running a prelim model creation. Only return cls
        return EfficientTransformer