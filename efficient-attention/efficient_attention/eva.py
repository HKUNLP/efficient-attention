import math
import warnings

import numpy as np
import torch
from efficient_attention import add_nested_argument
from efficient_attention.attn_utils import pad_to_multiple, prm_projection
from efficient_attention.local_attention import LocalAttention
from einops import rearrange
from torch import nn


# adapted from 
# https://github.com/lucidrains/FLASH-pytorch/blob/main/flash_pytorch/flash_pytorch.py#L54
class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale,
        num_heads,
        causal = False,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        causal = True,
        num_buckets = 32,
        max_distance = 128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        i, j, device = *x.shape[-2:], x.device
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        bias = self.relative_attention_bias(rp_bucket).permute([2, 0, 1]).unsqueeze(0).unsqueeze(2)
        return bias * self.scale



class EVA(LocalAttention):
    def __init__(self,
                 adaptive_proj='default',
                 num_landmarks=49,
                 use_t5_rpe=False,
                 *args,
                 **kwargs):
        super(EVA, self).__init__(*args, **kwargs)
        self.adaptive_proj = adaptive_proj
        if self.adaptive_proj in ['default']:
            self.adaptive_mu_q = nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim),
                nn.LayerNorm(self.head_dim),
            )
            self.adaptive_mu_k = nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim),
                nn.LayerNorm(self.head_dim),
            )
        elif self.adaptive_proj in ['no-ln']:
            self.adaptive_mu_q = nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim),
            )
            self.adaptive_mu_k = nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim),
            )
        elif self.adaptive_proj in ['none']:
            self.adaptive_mu_k = nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim),
                nn.LayerNorm(self.head_dim),
            )
        self.use_t5_rpe = use_t5_rpe
        self.num_landmarks = num_landmarks
        if self.use_rpe and not self.use_t5_rpe:
            warnings.warn(
                "By setting --use-rpe, the default relative positional embedding for local window is used."
                "We also implement a T5-style positional encoding, which we observe performs slightly better;"
                "This feature can be enabled by only setting --use-t5-rpe."
            )
        elif self.use_rpe and self.use_t5_rpe:
            raise NotImplementedError("Default RPE and T5-style RPE cannot be true simultaneously.")
        if self.use_t5_rpe:
            self.rel_pos_bias = T5RelativePositionBias(
                self.scale, 
                num_heads = self.num_heads,
                causal = False, 
                num_buckets=max(min(int((self.window_size + self.ext_size) / 2), 64), 16), 
                max_distance=self.window_size + self.ext_size
            )
        self.apply(self._init_weights)

    def _process_input(self, x, key_padding_mask):
        # this function re-implements the parent method.
        B, *seq_shape, C = x.shape
        N = np.prod(seq_shape)
        if self.attn_2d:
            assert len(seq_shape) == 2
            if self.window_size > 0:
                assert seq_shape[0] % self.window_size == 0 and seq_shape[1] % self.window_size == 0
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

    def forward(self, x, key_padding_mask = None):
        mask_val = -5e4
        ######################## Generate Proposal Parameters ###############################
        B, *seq_shape, C = x.shape
        orig_n = np.prod(seq_shape)
        x, key_padding_mask, seq_shape = self._process_input(x, key_padding_mask)
        N = np.prod(seq_shape)
        q, k, v = self.proj_and_split_heads(x)

        if key_padding_mask is None:
            key_padding_mask = torch.zeros(B, N, dtype=k.dtype, device=k.device)
        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool) # [b, 1, n, 1]
       
        w_q = self.window_partition(q, seq_shape, ext_window_size=0)
        w_k = self.window_partition(k, seq_shape, ext_window_size=self.ext_size)
        w_v = self.window_partition(v, seq_shape, ext_window_size=self.ext_size) # [b, h, w, j, d]

        if self.attn_2d:
            rf_win_size = int(math.sqrt(N // self.num_landmarks))
        else:
            rf_win_size = int(N // self.num_landmarks)
        # [b, h, c, j, d]
        rf_w_q = self.window_partition(q, seq_shape, window_size=rf_win_size, ext_window_size=self.ext_size)
        # [b, h, c, j, d]
        rf_w_k = self.window_partition(k, seq_shape, window_size=rf_win_size, ext_window_size=self.ext_size)
        # [b, h, c, j, d]
        rf_w_v = self.window_partition(v, seq_shape, window_size=rf_win_size, ext_window_size=self.ext_size)
        # compute local attention
        # [b, 1, c, j, 1]
        rf_w_mask = self.window_partition(
            key_padding_mask, 
            seq_shape, 
            window_size=rf_win_size,
            ext_window_size=self.ext_size,
            pad_val=1
            ).to(torch.bool)
        rf_w_q = rf_w_q.masked_fill(rf_w_mask, 0.)
        rf_w_k = rf_w_k.masked_fill(rf_w_mask, 0.)
        rf_w_v = rf_w_v.masked_fill(rf_w_mask, 0.)

        if self.adaptive_proj in ['default', 'no-ln']:
            rf_q_bar = self.adaptive_mu_q(rf_w_q.mean(dim=-2))
            rf_k_bar = self.adaptive_mu_k(rf_w_k.mean(dim=-2))
            # [b, h, c, d]
            mu = 0.5 * (rf_q_bar + rf_k_bar)
        elif self.adaptive_proj == 'none':
            rf_k_bar = self.adaptive_mu_k(rf_w_k.mean(dim=-2))
            mu = torch.zeros_like(rf_k_bar)
        ######################## Sampling from proposal ###############################
        if self.training:
            weights = mu + torch.randn_like(mu)
        else:
            weights = mu    
        # [b, h, c, j, d], [b, h, c, 1, d] -> [b, h, c, j]
        log_proj_w_k = prm_projection(rf_w_k, weights.unsqueeze(-2), normalize=False).squeeze(-2)
        log_proj_w_k = log_proj_w_k.masked_fill(rf_w_mask.squeeze(-1), mask_val)

        # [b, h, c, j] [b, h, c, j, d] -> [b, h, c, d]
        beta = torch.einsum('...cj,...cjd->...cd', torch.softmax(log_proj_w_k, dim=-1), rf_w_v)
        
        # compute approx. expectation of CVs.
        # [b, h, c, d]
        rfa_chunk = torch.einsum('...wid,...cd->...wic', w_q, self.scale * rf_k_bar)
        num_rfa_chunks = rfa_chunk.shape[-1]

        # compute local attention
        local_dots_mask = self.window_partition(
            key_padding_mask, 
            seq_shape, 
            ext_window_size=self.ext_size,
            pad_val=1
            ).to(torch.bool).transpose(-1, -2)

        log_qk_local_dot = torch.einsum('bhwie,bhwje->bhwij', w_q, w_k) * self.scale # [b, h, w, i, j]
        if self.use_t5_rpe:
            # here the t5-rpe-bias has already been scaled by \sqrt{d}
            log_qk_local_dot = log_qk_local_dot + self.rel_pos_bias(log_qk_local_dot)
        if self.use_rpe:
            log_qk_local_dot = self.add_rel_pos_bias(log_qk_local_dot)
        
        log_qk_local_dot = log_qk_local_dot.masked_fill(local_dots_mask, mask_val)
        local_len = log_qk_local_dot.shape[-1]
        
        # compute attention weights along with normalizing constant.
        attn = torch.softmax(torch.cat([log_qk_local_dot, rfa_chunk], dim=-1), dim=-1)
        local_attn, ra_attn = torch.split(attn, [local_len, num_rfa_chunks], dim=-1)
        output_local = torch.einsum('bhwij,bhwjd->bhwid', local_attn, w_v)
        output_snis = torch.einsum('bhwic,bhcd->bhwid', ra_attn, beta) 
        ######################## Combine them together ############################
        output = self.window_merge(output_snis + output_local, seq_shape) # [b, h, n, d]
        x = output.permute(0, 2, 1, 3).reshape((B,) + tuple(seq_shape) + (C,))
        x = self.proj(x)
        if orig_n is not None:
            x = x[..., :orig_n, :]
        x = self.proj_drop(x)
        return x

    @staticmethod
    def add_attn_specific_args(parent_parser, struct_name="attn_args", prefix=""):
        if hasattr(LocalAttention, "add_attn_specific_args"):
            parent_parser = LocalAttention.add_attn_specific_args(parent_parser, struct_name=struct_name, prefix=prefix)
        parser = parent_parser.add_argument_group("attention")
        _name_prefix = prefix + "-" if len(prefix) > 1 else ""
        add_nested_argument(parser, '--{}adaptive-proj'.format(_name_prefix), struct_name=struct_name, prefix=prefix, default='default', type=str)
        add_nested_argument(parser, '--{}num-landmarks'.format(_name_prefix), struct_name=struct_name, prefix=prefix, default=49, type=int)
        add_nested_argument(parser, '--{}use-t5-rpe'.format(_name_prefix), action='store_true', struct_name=struct_name, prefix=prefix, default=False)
        return parent_parser