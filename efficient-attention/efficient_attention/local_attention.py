import math
from typing import Optional, Tuple, Dict
from einops import rearrange

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from timm.models.layers import trunc_normal_
from efficient_attention import MultiheadAttention, add_nested_argument
from efficient_attention.attn_utils import (
    window_2d_partition, 
    window_2d_merge, 
    window_1d_partition,
    window_1d_merge,
    pad_to_multiple
    )

def default(val, d):
    return val if val is not None else d


# adapted from https://github.com/NVIDIA/transformer-ls/blob/master/lra/attention_transformer_ls.py
class LocalAttention(MultiheadAttention):

    def __init__(self,
                use_rpe=False, 
                window_size=2, 
                attn_2d=False,
                overlap_window=False,
                 *args,
                 **kwargs):
        super(LocalAttention, self).__init__(*args, **kwargs)
        self.window_size = window_size
        self.attn_2d = attn_2d
        self.use_rpe = use_rpe if window_size > 0 else False
        if overlap_window:
            self.ext_size = max(1, self.window_size // 2)
        else:
            self.ext_size = 0
        if self.use_rpe:
            if attn_2d:
                # handle the boarder conditions...
                w_pad = self.ext_size
                self.local_relative_position_bias_table = nn.Parameter(
                    torch.zeros(2 * (window_size + w_pad - 1) * (2 * w_pad + window_size + 1) + 1, self.num_heads))
                trunc_normal_(self.local_relative_position_bias_table, std=.02)

                # get pair-wise relative position index
                coords_h = torch.arange(-w_pad, w_pad + window_size)
                coords_w = torch.arange(-w_pad, w_pad + window_size)
                coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, 2w, 2w
                coords = coords.view(2, (window_size + w_pad * 2)**2).transpose(0, 1).unsqueeze(0) # 1, 4w**2, 2
                q_coords_hw = torch.arange(0, window_size)
                q_coords = torch.stack(torch.meshgrid([q_coords_hw, q_coords_hw])) # 2, w, w
                q_coords = q_coords.view(2, window_size**2).transpose(0, 1).unsqueeze(1) # w**2, 1, 2
                relative_coords = q_coords - coords
                relative_coords += w_pad + window_size - 1  # shift to start from 0
                relative_coords[:, :, 0] *= 2 * w_pad + window_size
                relative_position_index = relative_coords.sum(-1)  # w^2, 4w^2
                self.register_buffer("relative_position_index", relative_position_index)
            else:
                self.local_relative_position_bias_table = nn.Parameter(
                    torch.zeros(self.num_heads, window_size, window_size + self.ext_size * 2))
                trunc_normal_(self.local_relative_position_bias_table, std=.02)
        self.apply(self._init_weights)


    def add_rel_pos_bias(self, local_dots):
        if self.attn_2d:
            local_relative_position_bias = self.local_relative_position_bias_table[
                self.relative_position_index.view(-1)
                ].view(1, self.window_size * self.window_size, (self.ext_size*2 + self.window_size)**2, -1)  
            local_relative_position_bias = local_relative_position_bias.permute(
                0, 3, 1, 2).unsqueeze(2)
        else:
            local_relative_position_bias = self.local_relative_position_bias_table.unsqueeze(0).unsqueeze(2)
        return local_dots + local_relative_position_bias


    def window_partition(self, x, shape, ext_window_size, pad_val=0, window_size=None):
        window_size = default(window_size, self.window_size)
        if self.attn_2d:
            assert isinstance(shape, (list, tuple))
            H, W = shape
            return window_2d_partition(
                rearrange(x, '... (H W) d ->... H W d', H=H, W=W),
                window_size=window_size, 
                ext_window_size=ext_window_size, 
                pad_val=pad_val
                )
        else:
            return window_1d_partition(
                x, 
                window_size=window_size, 
                ext_window_size=ext_window_size, 
                pad_val=pad_val
                )
    
    def window_merge(self, x, shape, window_size=None):
        window_size = default(window_size, self.window_size)
        if self.attn_2d:
            assert isinstance(shape, (list, tuple))
            output = window_2d_merge(
                x,
                window_size=window_size, 
                hw_tuple=shape
                )
            return rearrange(output, '... H W d ->... (H W) d')
        else:
            return window_1d_merge(x)

    def _process_input(self, x, key_padding_mask):
        # this function is used in its children attention classes.
        B, *seq_shape, C = x.shape
        N = np.prod(seq_shape)
        if self.attn_2d:
            assert len(seq_shape) == 2
            if self.window_size > 0:
                assert seq_shape[0] % self.window_size == 0 and seq_shape[1] % self.window_size == 0
            x = x.reshape(B, N, C)
        else:
            if self.window_size > 0:
                if key_padding_mask is None:
                    x, key_padding_mask = pad_to_multiple(x, self.window_size, dim=-2, create_mask=True)
                else:
                    x = pad_to_multiple(x, self.window_size, dim=-2)
                    key_padding_mask = pad_to_multiple(key_padding_mask, self.window_size, dim=-1, value=True)
                N = x.shape[-2]
                seq_shape = [N]
        return x, key_padding_mask, seq_shape

    def _apply_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ):
        mask_val = -5e4
        if self.attn_2d:
            b, h, n, d = q.shape
            H = W = int(math.sqrt(n))
            shape = (H, W)
            assert H * W == n
            orig_n = n
        else:
            orig_n = q.shape[-2]
            if key_padding_mask is None:
                q, key_padding_mask = pad_to_multiple(q, self.window_size, dim=-2, create_mask=True)
            else:
                q = pad_to_multiple(q, self.window_size, dim=-2)
                key_padding_mask = pad_to_multiple(key_padding_mask, self.window_size, dim=-1, value=True)
            k, v = map(lambda t: pad_to_multiple(t, self.window_size, dim = -2), (k, v))
            b, h, n, d = q.shape
            shape = n
        if key_padding_mask is None:
            key_padding_mask = torch.zeros(b, n, dtype=q.dtype, device=q.device)
        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool) # [b, 1, n, 1]

        w_q = self.window_partition(q, shape, ext_window_size=0)
        w_k = self.window_partition(k, shape, ext_window_size=self.ext_size)
        w_v = self.window_partition(v, shape, ext_window_size=self.ext_size)
        local_dots = torch.einsum('bhwie,bhwje->bhwij', w_q, w_k) * self.scale # [b, h, w, i, j]

        if self.use_rpe:
            local_dots = self.add_rel_pos_bias(local_dots)

        local_dots_mask = self.window_partition(
            key_padding_mask, 
            shape, 
            ext_window_size=self.ext_size,
            pad_val=1
            ).to(torch.bool).transpose(-1, -2)
        local_dots.masked_fill_(local_dots_mask, mask_val)

        local_attn = local_dots.softmax(dim=-1)
        output = torch.einsum('bhwij,bhwje->bhwie', local_attn, w_v)

        output = self.window_merge(output, shape)[..., :orig_n, :]
        return output

    @staticmethod
    def add_attn_specific_args(parent_parser, struct_name="attn_args", prefix=""):
        if hasattr(super(LocalAttention, LocalAttention), "add_attn_specific_args"):
            parent_parser = super(LocalAttention, LocalAttention).add_attn_specific_args(parent_parser, struct_name=struct_name, prefix=prefix)
        parser = parent_parser.add_argument_group("Attention")
        _name_prefix = prefix + "-" if len(prefix) > 1 else ""
        add_nested_argument(parser, '--{}use-rpe'.format(_name_prefix), action='store_true', struct_name=struct_name, prefix=prefix, default=False)
        add_nested_argument(parser, '--{}window-size'.format(_name_prefix), struct_name=struct_name, prefix=prefix, default=4, type=int)
        add_nested_argument(parser, '--{}attn-2d'.format(_name_prefix), action='store_true', struct_name=struct_name, prefix=prefix, default=False)
        add_nested_argument(parser, '--{}overlap-window'.format(_name_prefix), action='store_true', struct_name=struct_name, prefix=prefix, default=False)
        return parent_parser
