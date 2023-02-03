import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
from einops import rearrange
from efficient_attention import AttentionFactory
__all__ = [
    'pvt_nano',
    'pvt_tiny', 
    'pvt_small', 
    'pvt_medium',
    'pvt_base',
    'pvt_large',
    'pvt_tiny2', 
    'pvt_small2', 
    'pvt_medium2',
    'pvt_base2',
    'pvt_large2',
    ]
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, 
        qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False, args=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        base_attn_args = {
                'dim': dim, 
                'num_heads': num_heads, 
                'qkv_bias': qkv_bias, 
                'attn_drop': attn_drop, 
                'proj_drop': proj_drop,
                }
        if sr_ratio > 1:
            attn_args = {
                **vars(args.attn_specific_args),
                **base_attn_args
            }
            if 'kernel_size' in attn_args:
                attn_args['kernel_size'] = sr_ratio
            self.attn_fn = AttentionFactory.build_attention(attn_name = args.attn_name, attn_args = attn_args)
        else:
            self.attn_fn = AttentionFactory.build_attention(attn_name = 'softmax', attn_args = base_attn_args)
    
    def forward(self, x):
        # B, N, C = x.shape
        # x = x.reshape(B, H, W, C)
        output = self.attn_fn(x)
        return output

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False, args=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear, args=args)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, use_conv_patchify=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        # GroupNorm(1, n) => LayerNorm
        # GroupNorm(n, n) => InstanceNorm
        if use_conv_patchify:
            self.proj = nn.Sequential(
                nn.Conv2d(3, embed_dim // 4, 3, 2, 1),
                nn.SyncBatchNorm(embed_dim // 4),
                nn.GELU(),
                nn.Conv2d(embed_dim // 4, embed_dim // 2, 3, 2, 1),
                nn.SyncBatchNorm(embed_dim // 2),
                nn.GELU(),
                nn.Conv2d(embed_dim // 2, embed_dim, 3, 1, 1),
                nn.SyncBatchNorm(embed_dim),
            )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)

        return x


class PyramidVisionTransformerV2(nn.Module):
    def __init__(self, args, linear=False):
        super().__init__()
        self.num_classes = args.num_classes
        self.depths = args.depths
        self.num_stages = 4

        dpr = [x.item() for x in torch.linspace(0, args.drop_path_rate, sum(args.depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(self.num_stages):
            patch_embed = OverlapPatchEmbed(img_size=args.input_size if i == 0 else args.input_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=args.in_chans if i == 0 else args.embed_dims[i - 1],
                                            embed_dim=args.embed_dims[i], use_conv_patchify=(args.use_conv_patchify and i == 0))

            block = nn.ModuleList([Block(
                dim=args.embed_dims[i], num_heads=args.num_heads[i], mlp_ratio=args.mlp_ratios[i], qkv_bias=args.qkv_bias, qk_scale=None,
                drop=args.drop_rate, attn_drop=args.attn_drop_rate, drop_path=dpr[cur + j], norm_layer=args.norm_layer,
                sr_ratio=args.sr_ratios[i], linear=linear, args=args)
                for j in range(args.depths[i])])
            norm = args.norm_layer(args.embed_dims[i])
            cur += args.depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(args.embed_dims[3], args.num_classes) if args.num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.permute(0, 3, 1, 2).contiguous()

        return x.mean(dim=(1,2))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Deit")
        # RFA parameters
        parser.add_argument('--use-conv-patchify', action='store_true', default=False)
        return parent_parser


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

def base_config(args):
    args.patch_size = 4
    args.in_chans = 3
    args.embed_dims=[64, 128, 320, 512]
    args.num_heads=[1, 2, 5, 8]
    args.mlp_ratios=[8, 8, 4, 4]
    args.qkv_bias=True,
    args.norm_layer=partial(nn.LayerNorm, eps=1e-6)
    args.sr_ratios=[8, 4, 2, 1]
    args.drop_path_rate = getattr(args, 'drop_path_rate', 0.1)
    return args


@register_model
def pvt_nano(args=None, **kwargs):
    if args:
        args = base_config(args)
        args.embed_dims=[32, 64, 160, 256]
        args.depths=[2, 2, 2, 2]
        args.num_heads=[1, 2, 5, 8]
        model = PyramidVisionTransformerV2(args)
        model.default_cfg = _cfg()
        return model
    else:
        # Possibly running a prelim model creation. Only return cls
        return PyramidVisionTransformerV2

@register_model
def pvt_tiny(args=None, **kwargs):
    if args:
        args = base_config(args)
        args.depths=[2, 2, 2, 2]
        model = PyramidVisionTransformerV2(args)
        model.default_cfg = _cfg()
        return model
    else:
        # Possibly running a prelim model creation. Only return cls
        return PyramidVisionTransformerV2


@register_model
def pvt_small(args=None, **kwargs):
    if args:
        args = base_config(args)
        args.depths=[3, 4, 6, 3]
        model = PyramidVisionTransformerV2(args)
        model.default_cfg = _cfg()
        return model
    else:
        # Possibly running a prelim model creation. Only return cls
        return PyramidVisionTransformerV2

# b3 
@register_model
def pvt_medium(args=None, **kwargs):
    if args:
        args = base_config(args)
        args.depths=[3, 4, 18, 3]
        args.drop_path_rate = 0.3
        args.clip_grad = 1.0
        model = PyramidVisionTransformerV2(args)
        model.default_cfg = _cfg()
        return model
    else:
        # Possibly running a prelim model creation. Only return cls
        return PyramidVisionTransformerV2
        
@register_model
def pvt_base(args=None, **kwargs):
    if args:
        args = base_config(args)
        args.depths=[3, 8, 27, 3]
        args.drop_path_rate = 0.3
        args.clip_grad = 1.0
        model = PyramidVisionTransformerV2(args)
        model.default_cfg = _cfg()
        return model
    else:
        # Possibly running a prelim model creation. Only return cls
        return PyramidVisionTransformerV2

@register_model
def pvt_large(args=None, **kwargs):
    if args:
        args = base_config(args)
        args.mlp_ratios=[4, 4, 4, 4]
        args.depths=[3, 6, 40, 3]
        args.drop_path_rate = 0.3
        args.clip_grad = 1.0
        model = PyramidVisionTransformerV2(args)
        model.default_cfg = _cfg()
        return model
    else:
        # Possibly running a prelim model creation. Only return cls
        return PyramidVisionTransformerV2

@register_model
def pvt_tiny2(args=None, **kwargs):
    if args:
        args = base_config(args)
        args.depths=[2, 2, 2, 2]
        args.num_heads=[2, 4, 10, 16]
        model = PyramidVisionTransformerV2(args)
        model.default_cfg = _cfg()
        return model
    else:
        # Possibly running a prelim model creation. Only return cls
        return PyramidVisionTransformerV2


@register_model
def pvt_small2(args=None, **kwargs):
    if args:
        args = base_config(args)
        args.depths=[3, 4, 6, 3]
        args.num_heads=[2, 4, 10, 16]
        model = PyramidVisionTransformerV2(args)
        model.default_cfg = _cfg()
        return model
    else:
        # Possibly running a prelim model creation. Only return cls
        return PyramidVisionTransformerV2

# b3 
@register_model
def pvt_medium2(args=None, **kwargs):
    if args:
        args = base_config(args)
        args.depths=[3, 4, 18, 3]
        args.num_heads=[2, 4, 10, 16]
        model = PyramidVisionTransformerV2(args)
        model.default_cfg = _cfg()
        return model
    else:
        # Possibly running a prelim model creation. Only return cls
        return PyramidVisionTransformerV2
        
@register_model
def pvt_base2(args=None, **kwargs):
    if args:
        args = base_config(args)
        args.depths=[3, 8, 27, 3]
        args.num_heads=[2, 4, 10, 16]
        model = PyramidVisionTransformerV2(args)
        model.default_cfg = _cfg()
        return model
    else:
        # Possibly running a prelim model creation. Only return cls
        return PyramidVisionTransformerV2

@register_model
def pvt_large2(args=None, **kwargs):
    if args:
        args = base_config(args)
        args.mlp_ratios=[4, 4, 4, 4]
        args.depths=[3, 6, 40, 3]
        args.num_heads=[2, 4, 10, 16]
        model = PyramidVisionTransformerV2(args)
        model.default_cfg = _cfg()
        return model
    else:
        # Possibly running a prelim model creation. Only return cls
        return PyramidVisionTransformerV2