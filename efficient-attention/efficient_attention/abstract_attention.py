import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from torch import Tensor
import numpy as np
from efficient_attention import add_nested_argument

class AbstractAttention(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super(AbstractAttention, self).__init__()
        self.name = f'{self.__class__.__name__}.{hash(self)}'

    def _reset_parameters(self):
        raise NotImplementedError

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_padding_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        need_head_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        static_kv: bool = False,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        **kwargs
    ) -> Tensor:
        # this member usually proceeds as follows:
        # 
        raise NotImplementedError

    # this member function is used to handle the internal computation among q, k and v.
    def _apply_attention(self, *args, **kwargs):
        raise NotImplementedError

class MultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads, fp32=False, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias
        self.fp32 = fp32

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
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
                
    def proj_and_split_heads(self, x):
        B, *seq_shape, C = x.shape
        N = np.prod(seq_shape)
        # x now has shape [b, n, c]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        return q, k, v

    def forward(self, x, key_padding_mask=None):
        B, *seq_shape, C = x.shape
        q, k, v = self.proj_and_split_heads(x)

        output = self._apply_attention(q, k, v, key_padding_mask)

        x = output.transpose(1, 2).reshape((B,) + tuple(seq_shape) + (C,))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def _apply_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        r"""
        Computes scaled dot product attention on query, key and value tensors, using
        an optional attention mask if passed, and applying dropout if a probability
        greater than 0.0 is specified.
        Returns a tensor pair containing attended values and attention weights.
        Args:
            q, k, v: query, key and value tensors. See Shape section for shape details.
            attn_mask: optional tensor containing mask values to be added to calculated
                attention. May be 2D or 3D; see Shape section for details.
        Shape:
            - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
                and E is embedding dimension.
            - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
                shape :math:`(Nt, Ns)`.
            - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
                have shape :math:`(B, Nt, Ns)`
        """
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        output = attn @ v
        return output

    @staticmethod
    def add_attn_specific_args(parent_parser, struct_name="attn_args", prefix=""):
        parser = parent_parser.add_argument_group("Attention")
        _name_prefix = prefix + "-" if len(prefix) > 1 else ""
        add_nested_argument(parser, '--{}fp32'.format(_name_prefix), struct_name=struct_name, prefix=prefix, default=False, action='store_true')
        return parent_parser