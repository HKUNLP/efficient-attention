import math
from typing import Callable, Optional, Tuple, Dict
from torch import nn
import torch
from torch import Tensor

from efficient_attention import MultiheadAttention, add_nested_argument
from efficient_attention.attn_utils import take_along_dim

# Randomized Attention in
# Linear Complexity Randomized Self-attention Mechanism
# see the paper https://arxiv.org/abs/2204.04667 for more details
class RandomizedAttention(MultiheadAttention):

    def __init__(self,
                num_samples=1,
                 *args,
                 **kwargs):
        super(RandomizedAttention, self).__init__(*args, **kwargs)
        self.num_samples = num_samples
        self.apply(self._init_weights)

    def _apply_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        b, h, n, d = q.shape
        if self.num_samples == 0:
            mu = q + k.mean(dim=-2, keepdim=True)
        else:
            pi = torch.softmax(torch.einsum('...nd,...md->...nm', self.scale * q, k), dim=-1) # b h lq lk
            if self.num_samples == -1:
                mu = q + torch.einsum('...nm,...md->...nd', pi, k)
            else:
                with torch.no_grad():
                    k_ind = torch.multinomial(pi.reshape(b * h * n, n), 1, replacement=True).reshape(b, h, n)
                k_prime = take_along_dim(k, k_ind, -2) # [b, h, n, d]
                mu = q + k_prime
        if self.training:
            weights = mu + torch.randn_like(mu)
        else:
            weights = mu
        data_dash = torch.einsum('...nd,...md->...nm', 
                                weights,
                                (self.scale * k),
                                ) # [b, h, lq, lk]
        norm = self.scale * torch.sum(k ** 2, dim=-1).unsqueeze(-2) / 2.0# [b, h, 1, lk]
        
        norm_k_sum = torch.softmax(data_dash - norm, dim=-1)  # [n, b, h, l_c, l_k]
        output = torch.einsum('...nm,...md->...nd', norm_k_sum, v) # [b, h, c, d]
        return output

    @staticmethod
    def add_attn_specific_args(parent_parser, struct_name="attn_args", prefix=""):
        if hasattr(super(RandomizedAttention, RandomizedAttention), "add_attn_specific_args"):
            parent_parser = super(RandomizedAttention, RandomizedAttention).add_attn_specific_args(parent_parser, struct_name=struct_name, prefix=prefix)
        parser = parent_parser.add_argument_group("Attention")
        _name_prefix = prefix + "-" if len(prefix) > 1 else ""
        # RA parameters
        add_nested_argument(parser, '--{}num-samples'.format(_name_prefix), struct_name=struct_name, prefix=prefix, default=1, type=int,
                                help='number of random features')
        return parent_parser
