
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import trunc_normal_
from efficient_attention import MultiheadAttention, add_nested_argument
from efficient_attention.attn_utils import (
     FlattenTranspose, hyperm_projection, prm_projection
)

def default(val, d):
    return val if val is not None else d

class LinearRA(MultiheadAttention):
    def __init__(self, 
                num_landmarks=49,
                kernel_size=None,
                proposal_gen='pool',
                use_antithetics=False,
                use_multisample=False,
                pool_module_type='light',
                mis_type='mis-opt',
                alpha_coeff=1.0,
                *args,
                **kwargs):
        super(LinearRA, self).__init__(*args, **kwargs)
        self.num_landmarks = num_landmarks
        self.proposal_gen = proposal_gen
        self.use_antithetics = use_antithetics
        self.use_multisample = use_multisample

        self.pool_module_type = pool_module_type
        self.mis_type = mis_type
        self.alpha_coeff = alpha_coeff

        if self.pool_module_type == 'dense':
            num_channels = self.dim
        elif self.pool_module_type == 'light':
            num_channels = self.head_dim

        if self.proposal_gen.startswith('pool'):
            output_size = int(math.sqrt(self.num_landmarks))
            self.q_bar_gen = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size),
                FlattenTranspose(),
                nn.Linear(num_channels, num_channels),
                nn.LayerNorm(num_channels), 
                )
            self.k_bar_gen = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size),
                FlattenTranspose(),
                nn.Linear(num_channels, num_channels),
                nn.LayerNorm(num_channels), 
                )
        elif self.proposal_gen.startswith('adaptive-1d'):
            self.q_bar_gen = nn.Sequential(
                nn.Linear(num_channels, num_channels),
                nn.LayerNorm(num_channels),
            )
            self.k_bar_gen = nn.Sequential(
                nn.Linear(num_channels, num_channels),
                nn.LayerNorm(num_channels),
            )
        else:
            raise NotImplementedError
        self.apply(self._init_weights)

    def _proposal_gen_1d(self, x, key_padding_mask=None):
        landmarks = self.num_landmarks
        q, k, v = self.proj_and_split_heads(x)
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(-1).type_as(v)
            q = q * (1. - mask)
            k = k * (1. - mask)
            v = v * (1. - mask)
        b, h, n, d = q.shape # [b, num_heads, N, D]
        # if self.proposal_gen == 'ortho':
        #     k_bar = k.mean(-2, keepdim=True)
        #     with torch.no_grad():
        #         q_bar = orthogonal_landmarks(q, k, landmarks)
        #     # diag = False
        segs = n // landmarks
        if self.proposal_gen.startswith('adaptive-1d'):
            q2 = self.q_bar_gen(q)
            k2 = self.k_bar_gen(k)
        else:
            q2 = q
            k2 = k
        if n <= landmarks:
            q_bar = q2
            k_bar = k2
        elif (n % landmarks == 0):
            q_bar = q2.reshape(b, h, landmarks, n // landmarks, d).mean(dim = -2)
            k_bar = k2.reshape(b, h, landmarks, n // landmarks, d).mean(dim = -2)
        else:
            num_k = (segs + 1) * landmarks - n

            keys_landmarks_f = k2[:, :, :num_k * segs, :].reshape(
                b, h, num_k, segs, d).mean(dim = -2)
            keys_landmarks_l = k2[:, :, num_k * segs:, :].reshape(
                b, h, landmarks - num_k, segs + 1, d).mean(dim = -2)
            k_bar = torch.cat((keys_landmarks_f, keys_landmarks_l), dim = -2)

            queries_landmarks_f = q2[:, :, :num_k * segs, :].reshape(
                b, h, num_k, segs, d).mean(dim = -2)
            queries_landmarks_l = q2[:, :, num_k * segs:, :].reshape(
                b, h, landmarks - num_k, segs + 1, d).mean(dim = -2)
            q_bar = torch.cat((queries_landmarks_f, queries_landmarks_l), dim = -2)


        return q_bar, k_bar, q, k, v

    def _proposal_gen_2d(self, x, key_padding_mask=None):
        b, H, W, c = x.shape
        if self.pool_module_type == 'dense':
            qkv = self.qkv(x).reshape((b, H * W, 3, self.num_heads, self.head_dim)).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2] # [b, h, H * W, d]

            q_bar = self.q_bar_gen(q.transpose(-1,-2).reshape(b, c, H, W))
            k_bar = self.k_bar_gen(k.transpose(-1,-2).reshape(b, c, H, W))

            q_bar = q_bar.reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k_bar = k_bar.reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        elif self.pool_module_type == 'light':
            qkv = self.qkv(x).reshape((b, H * W, 3, self.num_heads, self.head_dim)).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2] # [b, h, H * W, d]
            assert q.dim() == 4 and k.dim() == 4
            temp_q = q.reshape(b * self.num_heads, H, W, self.head_dim).permute(0, 3, 1, 2)
            temp_k = k.reshape(b * self.num_heads, H, W, self.head_dim).permute(0, 3, 1, 2)

            q_bar = self.q_bar_gen(temp_q)
            k_bar = self.k_bar_gen(temp_k)
            q_bar = q_bar.reshape(b, self.num_heads, -1, self.head_dim)
            k_bar = k_bar.reshape(b, self.num_heads, -1, self.head_dim)

        else:
            raise NotImplementedError


        if self.proposal_gen.endswith('mixed'):
            k_logits = torch.einsum('...pd, ...cd-> ...pc', self.scale * k_bar, k_bar)
            # To mix information over key landmarks, 
            # we could also incorporate the value information into the weights;
            # This approximates the optimal proposal distribution for our importance
            # sampling estimate. However, it brings marginal benefits for most cases.
            if self.proposal_gen.endswith('-vmixed'):
                v_bar = F.adaptive_avg_pool2d(
                    v.reshape(b * self.num_heads, H, W, self.head_dim).permute(0, 3, 1, 2), 
                    int(math.sqrt(self.num_landmarks))
                    ).reshape(b, self.num_heads, self.head_dim, -1).transpose(-2, -1) # [b, h, c, d]
                log_v_norm = torch.log(torch.linalg.vector_norm(v_bar, ord=2, dim=-1) + 1e-4).unsqueeze(-2) # b h c 1
                k_logits = k_logits + log_v_norm
            k_bar = torch.einsum(
                '...pc, ...cd->...pd',
                torch.softmax(k_logits, dim=-1),
                k_bar
            )
        return q_bar, k_bar, q, k, v

    def forward(self, x, key_padding_mask=None):
        ######################## Generate Proposal Parameters ###############################
        B, *seq_shape, C = x.shape
        if len(seq_shape) == 2:
            q_bar, k_bar, q, k, v = self._proposal_gen_2d(x, key_padding_mask)
            mu = q_bar + k_bar
        elif len(seq_shape) == 1:
            q_bar, k_bar, q, k, v = self._proposal_gen_1d(x, key_padding_mask)
            mu = q_bar + k_bar

        ######################## Sampling from proposal ###############################
        if self.training:
            if self.use_multisample:
                noise = torch.randn(B, self.num_heads, mu.shape[-2] * 2, self.head_dim, dtype=mu.dtype, device=mu.device)
                weights = mu.repeat(1, 1, 2, 1) + noise
            elif self.use_antithetics:
                noise = torch.randn_like(mu)
                weights = torch.cat([mu + noise, mu - noise], dim=-2)
            else:
                weights = mu + torch.randn_like(mu)
        else:
            weights = mu
        
        ######################## Computing SNIS estimates ###############################
        log_proj_q = prm_projection(q, weights, normalize=False) # [b, h, c, lq]
        log_proj_k = prm_projection(k, weights, normalize=False) # [b, h, c, lk]

        if key_padding_mask is not None:
            log_proj_k = log_proj_k.masked_fill(
                        key_padding_mask.unsqueeze(1).unsqueeze(-2).to(torch.bool),
                        float("-inf"),
                    )

        # computes f(\omega), where no proj_q is involved since it is cancelled out at both numerator and denominator.
        kv_stats = torch.einsum('...cm,...md->...cd', torch.softmax(log_proj_k, dim=-1), v) # [b, h, c, d]

        ## Different methods differ in computing the proposal density & \alpha's
        if self.mis_type == 'mis-biased':
            log_proj_mu = prm_projection(mu, weights, normalize=False) # [b, h, c, c_mu]
            log_alpha = torch.einsum('...cd,...nd->...cn', self.scale * mu, q) # [b,h,c,l_q]
            if self.training:
                if self.use_multisample or self.use_antithetics:
                    log_alpha = log_alpha.repeat(1, 1, 2, 1)
            log_proposal = torch.logsumexp(log_proj_mu, dim=-1, keepdim=True) #[b,h,c,lq]
        elif self.mis_type == 'mis-opt':
            log_tnc = torch.einsum('...cd,...nd->...cn', self.scale * q_bar, q)
            t_nc = torch.softmax(log_tnc, dim=-1)# [b,h,c,l_q]
            if self.training:
                if self.use_multisample or self.use_antithetics:
                    mu = mu.repeat(1, 1, 2, 1)
                    t_nc = t_nc.repeat(1, 1, 2, 1)
            log_proj_mu = prm_projection(mu, weights, normalize=False) # [b, h, c, c_mu]
            log_proposal = torch.diagonal(log_proj_mu, dim1=-1, dim2=-2).unsqueeze(-1)
            balanced_heuristics = torch.exp(log_proposal - torch.logsumexp(log_proj_mu, dim=-1, keepdim=True))
            alpha_prev = balanced_heuristics + self.alpha_coeff * (t_nc - t_nc.mean(-2, keepdim=True))
            log_alpha = torch.log(alpha_prev.clamp(min=1e-8)) # [b, h, c, l_q]
        elif self.mis_type == 'mis-bh':
            log_proj_mu = prm_projection(mu, weights, normalize=False) # [b, h, c, c_mu]
            log_alpha = 0.
            log_proposal = torch.logsumexp(log_proj_mu, dim=-1, keepdim=True) # [b, h, c, 1]
        else:
            raise NotImplementedError("The attn_type {} is not supported yet.".format(self.mis_type))
        
        # computes self-normalized importance weights.
        log_true_prob = log_proj_q + torch.logsumexp(log_proj_k, dim=-1, keepdim=True)
        log_iw_ratio = log_alpha + log_true_prob - log_proposal #[b,h,c,lq]
        sniw = torch.softmax(log_iw_ratio, dim=-2)# [b, h, c, lq]

        # the sum over all samples to compute the final estimator.
        output = torch.einsum('...cn, ...cd->...nd', sniw, kv_stats)

        x = output.transpose(1, 2).reshape((B,) + tuple(seq_shape) + (C,))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def add_attn_specific_args(parent_parser, struct_name="attn_args", prefix=""):
        if hasattr(super(LinearRA, LinearRA), "add_attn_specific_args"):
            parent_parser = super(LinearRA, LinearRA).add_attn_specific_args(parent_parser, struct_name=struct_name, prefix=prefix)
        parser = parent_parser.add_argument_group("attention")
        _name_prefix = prefix + "-" if len(prefix) > 1 else ""
        # LARA parameters
        add_nested_argument(parser, '--{}num-landmarks'.format(_name_prefix), struct_name=struct_name, prefix=prefix, default=49, type=int)
        add_nested_argument(parser, '--{}kernel-size'.format(_name_prefix), struct_name=struct_name, prefix=prefix, default=None, type=int)
        add_nested_argument(parser, '--{}pool-module-type'.format(_name_prefix), struct_name=struct_name, prefix=prefix, default='light', type=str)
        add_nested_argument(parser, '--{}mis-type'.format(_name_prefix), struct_name=struct_name, prefix=prefix, default='mis-opt', type=str)
        add_nested_argument(parser, '--{}proposal-gen'.format(_name_prefix), struct_name=struct_name, prefix=prefix, default='pool', type=str)
        add_nested_argument(parser, '--{}use-antithetics'.format(_name_prefix), struct_name=struct_name, prefix=prefix, action='store_true', default=False)
        add_nested_argument(parser, '--{}use-multisample'.format(_name_prefix), struct_name=struct_name, prefix=prefix, action='store_true', default=False)
        add_nested_argument(parser, '--{}alpha-coeff'.format(_name_prefix), struct_name=struct_name, prefix=prefix, default=1.0, type=float)
        return parent_parser