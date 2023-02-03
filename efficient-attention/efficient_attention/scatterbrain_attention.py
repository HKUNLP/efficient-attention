import math
from typing import Callable, Optional, Tuple, Dict
from einops import rearrange
from torch import nn
import torch
from torch import Tensor
from functools import partial
import numpy as np
from efficient_attention import add_nested_argument
from efficient_attention.local_attention import LocalAttention
from efficient_attention.kernelized_attention import (
    KernelizedAttention, 
    create_proj_matrix)
from efficient_attention.attn_utils import log_add_exp

def log_favorp_projection_for_scatterbrain(
        data: torch.Tensor,
        projection_matrix: torch.Tensor,
        is_query: bool,
        eps: float=0.0001):
    """
    Constructs nonnegative kernel features for fast softmax attention.
    Args:
      data: input for which features are computes
      projection_matrix: random matrix used to compute features
      batch_dims_t: tuple of batch dimensions
      is_query: predicate indicating whether input data corresponds to queries or
        keys
      eps: numerical stabilizer.
    Returns:
      Random features for fast softmax attention.
    """
    # We have e^{qk^T/sqrt{d}} = e^{q_norm k_norm^T}, where
    # w_norm = w * data_normalizer for w in {q,k}.
    data_normalizer = (data.shape[-1] ** -0.25)
    ratio = projection_matrix.shape[1]
    data_dash = torch.einsum('bn...d,njd->bn...j', 
                            (data_normalizer * data),
                            projection_matrix)
    diag_data = torch.sum(data ** 2, dim=-1)
    diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
    diag_data = diag_data.unsqueeze(-1)


    # adapted from HazyResearch/pixelfly/blob/master/src/models/modules/attention/feature_maps_sb.py#L85
    if is_query:
        # data_dash_log = data_dash - torch.amax(data_dash, dim=-1, keepdim=True)
        data_dash = data_dash - diag_data - math.log(ratio) / 2
    else:
        data_dash = data_dash - diag_data - math.log(ratio) / 2
    return data_dash

# a re-implementation of 
# Scatterbrain: Unifying Sparse and Low-rank Attention Approximation
# see the paper https://arxiv.org/abs/2110.15343 for more details
class ScatterBrain(KernelizedAttention, LocalAttention):
    def __init__(self,
                 *args,
                 **kwargs):
        super(ScatterBrain, self).__init__(*args, **kwargs)
        # otherwise, there is not an easy way to account for the
        # relative bias from the Performer side.
        self.apply(self._init_weights)

    def q_k_projection(self, q, k, random_proj=None):
        # rewrite the favorp projection.
        if self.proj_method == 'favorp':
            assert random_proj is not None
            # feature_proj = partial(favorp_projection_for_scatterbrain, projection_matrix = random_proj)
            feature_proj = partial(log_favorp_projection_for_scatterbrain, projection_matrix = random_proj)
            q = feature_proj(q, is_query = True)
            k = feature_proj(k, is_query = False)
            return q, k
        else:
            return super(ScatterBrain, self).q_k_projection(q, k, random_proj=random_proj)
    
    def forward(self, x, key_padding_mask = None):
        B, *seq_shape, C = x.shape
        orig_n = np.prod(seq_shape)
        x, key_padding_mask, seq_shape = self._process_input(x, key_padding_mask)
        N = np.prod(seq_shape)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if key_padding_mask is None:
            key_padding_mask = torch.zeros(B, N, dtype=k.dtype, device=k.device)
        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool) # [b, 1, n, 1]

        # Low-rank attention
        projection_matrix = self.get_proj_matrix(device=q.device, dtype=q.dtype)

        if self.proj_method == 'favorp':
            log_proj_q, log_proj_k = self.q_k_projection(q, k, projection_matrix)
        else:
            q_prime, k_prime = self.q_k_projection(q, k, projection_matrix)

        if key_padding_mask is not None:
            log_proj_k = log_proj_k.masked_fill(key_padding_mask, float('-inf'))


        w_q = self.window_partition(q, seq_shape, ext_window_size=0)
        w_k = self.window_partition(k, seq_shape, ext_window_size=self.ext_size)
        w_v = self.window_partition(v, seq_shape, ext_window_size=self.ext_size)


        ##################### Compute RA local statistics #####################
        # [b, h, g, w, c]
        w_log_proj_q = self.window_partition(log_proj_q, seq_shape, ext_window_size=0)
        w_log_proj_k = self.window_partition(log_proj_k, seq_shape, ext_window_size=self.ext_size)
        
        # [b, h, 1, lk, c]
        log_proj_k = log_proj_k.unsqueeze(-3)
        # the taken axis for w_log_proj_k is (-1, -3) so that
        # the resulting shape of max_proj_k will not get broadcasted to
        # (b, h, g, c, 1); otherwise, proj_k will have shape (b, h, g, c, lk)
        # shape [b, h, 1, 1, c]
        max_proj_k = torch.maximum(
            torch.amax(log_proj_k, dim=-2, keepdim=True).detach() , 
            torch.amax(w_log_proj_k, dim=(-2, -3), keepdim=True).detach()
            )
        proj_k = torch.exp(log_proj_k - max_proj_k) # [b, h, 1, lk, c]
        w_proj_k = torch.exp(w_log_proj_k - max_proj_k) # [b, h, g, w, c]
        # [b, h, g, c, d]
        # # computes f(\omega), where no proj_q is involved since it is cancelled out at both numerator and denominator.
        # # compute nonlocal contexts
        # [b,h,g,c,d] / [b,h,1,c,1]
        kv_stats = (
            torch.einsum('bhtmc,bhmd->bhtcd', proj_k, v) - 
            torch.einsum('bhgwc,bhgwd->bhgcd', w_proj_k, w_v)
        ) / (torch.sum(proj_k, dim=-2) - torch.sum(w_proj_k, dim=-2)).unsqueeze(-1).clamp(min=1e-3)

        # compute global linearized contexts
        log_sum_proj_k = torch.logsumexp(log_proj_k, dim=-2, keepdim=True) # [b, h, 1, 1, c]
        # compute local linearized contexts
        log_sum_proj_k_local = torch.logsumexp(w_log_proj_k, dim=-2, keepdim=True) # [b, h, g, 1, c]
        # compute nonlocal linearized contexts
        log_sum_proj_k_nonlocal = log_add_exp(log_sum_proj_k, log_sum_proj_k_local, mask=(1, -1))

        # [b, h, g, c, w] [b, h, g, c, 1] [b, h, 1, c, 1] 
        # computes self-normalized importance weights.
        log_rfa_d = w_log_proj_q + log_sum_proj_k_nonlocal #[b,h,g,w,c]

        local_dots_mask = self.window_partition(
            key_padding_mask, 
            seq_shape, 
            ext_window_size=self.ext_size,
            pad_val=1
            ).to(torch.bool).transpose(-1, -2)

        log_qk_local_dot = torch.einsum('bhwie,bhwje->bhwij', w_q, w_k) * self.scale # [b, h, w, i, j]
                
        if self.use_rpe:
            log_qk_local_dot = self.add_rel_pos_bias(log_qk_local_dot)
        log_qk_local_dot = log_qk_local_dot.masked_fill(local_dots_mask, float('-inf'))
        local_len = log_qk_local_dot.shape[-1]
        lara_len = log_rfa_d.shape[-1]

        # print(log_qk_local_dot.shape, log_lara_d.shape)
        attn = torch.softmax(torch.cat([log_qk_local_dot, log_rfa_d], dim=-1), dim=-1)
        local_attn, rfa_attn = torch.split(attn, [local_len, lara_len], dim=-1)
        output_local = torch.einsum('bhwij,bhwje->bhwie', local_attn, w_v) 
        output_snis = torch.einsum('bhwic,bhwce->bhwie', rfa_attn, kv_stats) 
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
        parser = parent_parser.add_argument_group("Attention")
        _name_prefix = prefix + "-" if len(prefix) > 1 else ""
        add_nested_argument(parser, '--{}approx-attn-dim'.format(_name_prefix), struct_name=struct_name, prefix=prefix, default=64, type=int,
                            help='number of random features')
        add_nested_argument(parser, '--{}proj-method'.format(_name_prefix), struct_name=struct_name, prefix=prefix, default='favorp', type=str,
                            help='which attention method is used for RFA')
        add_nested_argument(parser, '--{}cos-weighting'.format(_name_prefix), struct_name=struct_name, prefix=prefix, action='store_true', default=False, help='')
        add_nested_argument(parser, '--{}sample-scheme'.format(_name_prefix), struct_name=struct_name, prefix=prefix, default='default', type=str)
        return parent_parser
