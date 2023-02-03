import math
from typing import Callable, Optional, Tuple, Dict
from torch import nn
import torch
from torch import Tensor
from functools import partial

from efficient_attention import MultiheadAttention, add_nested_argument


############# available feature projections ######################

def dpfp_projection(x, is_query=True, nu=1):
    x = torch.cat([torch.relu(x), torch.relu(-x)], dim=-1)
    x_rolled = torch.cat([x.roll(shifts=j, dims=-1)
                    for j in range(1, nu+1)], dim=-1)
    x_repeat = torch.cat([x] * nu, dim=-1)
    return x_repeat * x_rolled

def favorp_projection(
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
    ratio = (projection_matrix.shape[1] ** -0.5)
    data_dash = torch.einsum('bh...d,hjd->bh...j', 
                            (data_normalizer * data),
                            projection_matrix)
    diag_data = torch.sum(data ** 2, dim=-1)
    diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
    diag_data = diag_data.unsqueeze(-1)
    
    if is_query:
        data_dash_log = data_dash - diag_data
        stabilizer = torch.amax(data_dash, dim=-1, keepdim=True).detach()
        data_dash = ratio * torch.exp(data_dash_log - stabilizer) + eps
    else:
        data_dash_log = data_dash - diag_data
        stabilizer = torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()
        data_dash = ratio * torch.exp(data_dash_log - stabilizer) + eps
    return data_dash

def fourier_projection(
        data: torch.Tensor,
        projection_matrix: torch.Tensor,
        is_query=None,
        eps: float=0.0001):
    # x: [seq_len, bsz, num_heads, head_dim]
    # random_matrices: [num_heads, proj_dim, head_dim]

    # [1, 1, num_heads, 1]
    # data_normalizer is used to recover scaled attn.
    data_normalizer = (data.shape[-1] ** -0.25)
    data_dash = torch.einsum('bn...d,njd->bn...j', 
                        data * data_normalizer,
                        projection_matrix)

    # [bsz, num_heads, len, 2 * proj_dim]
    ratio = projection_matrix.shape[1] ** -0.5
    phi_data = ratio * torch.cat([torch.sin(data_dash), torch.cos(data_dash)], dim=-1)


    h_data = torch.sum(data ** 2, dim=-1)
    h_data = (h_data / 2.0) * data_normalizer * data_normalizer
    # here the last dimension is the length (since it is before unsqueeze)
    h_data = torch.exp(h_data - torch.amax(h_data, dim=-1, keepdim=True).detach()).unsqueeze(-1)

    phi_data_prime = h_data * phi_data

    return phi_data_prime

def nonlinear_map(data: Tensor, mapping_fn: Callable, is_query: Optional[bool] = False, eps=1e-1):
    return mapping_fn(data) + eps

def generalized_projection(
        data: torch.Tensor,
        projection_matrix: torch.Tensor,
        is_query: bool,
        projection_fn: Callable,
        eps: float=0.001):
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
    del is_query
    ratio = projection_matrix.shape[1] ** -0.5
    data_normalizer = (data.shape[-1] ** -0.25)
    data_dash = ratio * torch.einsum('bn...d,njd->bn...j', 
                        data * data_normalizer,
                        projection_matrix)
    return projection_fn(data_dash) + eps

def linear_attention(q_prime, k_prime, v, eps=1e-2):
    kv = torch.einsum('...nm,...nd->...md', k_prime, v)
    qkv = torch.einsum('...nm,...md->...nd', q_prime, kv)
    normalizer = torch.einsum('...nm,...m->...n', q_prime, k_prime.sum(dim=-2))
    output = qkv / normalizer.unsqueeze(-1).clamp(min=eps)
    return output

def cos_reweighted_linear_attention(q_prime, k_prime, v, lengths=None, eps=1e-2):
    # lengths -> [batch_size]
    if lengths is None:
        b, max_len, dtype, device = v.shape[0], v.shape[-2], v.dtype, v.device
        # For each sample x in the batch, calculate M(x) = len(x)
        M = (1.0 / max_len) * torch.ones(b, dtype=dtype, device=device)
    else:
        M = lengths
    # M -> [batch_size]
    idxs = math.pi / 2 * torch.arange(max_len, dtype=dtype, device=device)
    # idxs -> [max_len]
    idxs = torch.outer(M, idxs)  # [..., None, None]
    # idxs -> [batch_size, max_len]

    cos = torch.cos(idxs).unsqueeze(-1).unsqueeze(1).detach() # [b, 1, n]
    sin = torch.sin(idxs).unsqueeze(-1).unsqueeze(1).detach()

    # cos, sin -> [batch_size, seq_len]
    q_cos = q_prime * cos
    q_sin = q_prime * sin
    k_cos = k_prime * cos
    k_sin = k_prime * sin

    kv_cos = torch.einsum('...nm,...nd->...md', k_cos, v)
    kv_sin = torch.einsum('...nm,...nd->...md', k_sin, v)

    qkv_cos = torch.einsum('...nm,...md->...nd', q_cos, kv_cos)
    qkv_sin = torch.einsum('...nm,...md->...nd', q_sin, kv_sin)

    normalizer_cos = torch.einsum('...nm,...m->...n', q_cos, k_cos.sum(dim=-2))
    normalizer_sin = torch.einsum('...nm,...m->...n', q_sin, k_sin.sum(dim=-2))

    output = (qkv_cos + qkv_sin) / (normalizer_cos + normalizer_sin).unsqueeze(-1).clamp(min=eps)
    return output


class DeterministicLearnableFourierFeatures(nn.Module):
    def __init__(self, num_heads, dim, fourier_dim, std=0.02):
        super().__init__()
        self.dim = dim
        self.random_proj = nn.Parameter(std * torch.randn(num_heads, fourier_dim // 2, dim))
        self.phi = nn.Sequential(
            nn.Linear(fourier_dim, fourier_dim),
            nn.ReLU(),
        )
        ## version 1:
        # self.phi = nn.Sequential(
        #     nn.Linear(fourier_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, fourier_dim),
        #     nn.ReLU(),
        # )
    def forward(self, x, random_proj=None, is_query=False):
        # x has shape [b, n, ..., d]
        assert x.shape[-1] == self.dim

        # x_norm = torch.sum(x ** 2, dim=-1) * (self.dim ** -0.5)
        # x_norm = torch.exp(x_norm - torch.max(x_norm, dim=-1, keepdim=True)[0]).unsqueeze(-1)
        projected_x = torch.einsum('bn...d,njd->bn...j', x, self.random_proj)
        fourier_feature = torch.cat([projected_x.cos(), projected_x.sin()], dim=-1)
        return self.phi(fourier_feature * (self.dim ** -0.5))

def orthogonal_matrix_chunk(cols, device = None, dtype=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t().to(dtype)

def create_proj_matrix(num_heads, proj_dim, input_dim, ortho=False, seed=0, device=None, dtype=None):
    if ortho:
        return torch.stack(
            [
                gaussian_orthogonal_random_matrix(proj_dim, input_dim, seed=seed + h * 1000, device=device, dtype=dtype)
                for h in range(num_heads)
            ], dim=0)
    else:
        return torch.randn(num_heads, proj_dim, input_dim, device=device, dtype=dtype)

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, seed=0, device=None, dtype=None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []
    cur_seed = seed

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device, dtype=dtype)
        block_list.append(q)
        cur_seed = cur_seed + 1

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device, dtype=dtype)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    multiplier = torch.randn((nb_rows, nb_columns), device=device, dtype=dtype).norm(dim=1)

    return torch.diag(multiplier) @ final_matrix

class KernelizedAttention(MultiheadAttention):

    def __init__(self,
                 approx_attn_dim=64,
                 proj_method='favorp',
                 cos_weighting=False,
                 sample_scheme='default',
                 *args,
                 **kwargs):
        super(KernelizedAttention, self).__init__(*args, **kwargs)
        self.approx_attn_dim = approx_attn_dim
        self.proj_method = proj_method
        self.cos_weighting = cos_weighting # use cosformer or not
        self.sample_scheme = sample_scheme

        self.use_random_proj = False
        if self.proj_method == 'dpfp':
            self.use_random_proj = False
            self.nu = (self.approx_attn_dim // self.head_dim) // 2
            assert self.nu > 0 and isinstance(self.nu, int), "approx_attn_dim must be a multiple of 2*head_dim!"
            self.feature_proj = partial(dpfp_projection, nu = self.nu)

        elif self.proj_method == 'mlp-fourier':
            self.use_random_proj = False
            self.feature_proj = DeterministicLearnableFourierFeatures(self.num_heads, self.head_dim, self.approx_attn_dim)
        
        elif self.proj_method in ['favorp', 'relu', 'fourier']:
            self.use_random_proj = True
            if self.sample_scheme == 'default':
                self.register_buffer('eval_proj', create_proj_matrix(
                    self.num_heads, 
                    self.approx_attn_dim, 
                    self.head_dim, 
                    ortho=True
                    )
                )
            elif self.sample_scheme == 'fixed':
                self.register_buffer('random_proj', create_proj_matrix(
                    self.num_heads, 
                    self.approx_attn_dim, 
                    self.head_dim, 
                    ortho=True
                    )
                )
            elif self.sample_scheme == 'learnable':
                self.random_proj = nn.Parameter(
                    create_proj_matrix(
                        self.num_heads, 
                        self.approx_attn_dim, 
                        self.head_dim, 
                        ortho=True
                    )
                )
            else:
                raise NotImplementedError('other sample schemes are not implemented yet.')


        elif self.proj_method in ['relu-only', 'sigmoid-only']:
            self.use_random_proj = False
            mapping_fn = getattr(torch, self.proj_method.split('-')[0])
            self.feature_proj = partial(nonlinear_map, mapping_fn=mapping_fn)
        else:
            raise NotImplementedError    
        self.apply(self._init_weights)
        
    def q_k_projection(self, q, k, random_proj=None):
        if self.proj_method == 'favorp':
            assert random_proj is not None
            feature_proj = partial(favorp_projection, projection_matrix = random_proj)
        elif self.proj_method == 'fourier':
            assert random_proj is not None
            feature_proj = partial(fourier_projection, projection_matrix = random_proj)
        elif self.proj_method == 'relu':
            assert random_proj is not None
            feature_proj = partial(generalized_projection, projection_matrix = random_proj, projection_fn=torch.relu)
        else:
            assert random_proj is None and hasattr(self, 'feature_proj')
            feature_proj = self.feature_proj
        q = feature_proj(q, is_query = True)
        k = feature_proj(k, is_query = False)
        return q, k

    def _linear_attention(self, q_prime, k_prime, v):
        if self.cos_weighting:
            output = cos_reweighted_linear_attention(q_prime, k_prime, v)
        else:
            output = linear_attention(q_prime, k_prime, v)
        return output
    
    def get_proj_matrix(self, device=None, dtype=None):
        if self.use_random_proj:
            if self.sample_scheme == 'default':
                if self.training:
                    projection_matrix = create_proj_matrix(
                        self.num_heads, self.approx_attn_dim, self.head_dim, ortho=False, device=device, dtype=dtype)
                else:
                    projection_matrix = self.eval_proj
            elif self.sample_scheme in ['fixed', 'learnable']:
                projection_matrix = self.random_proj
        else:
            projection_matrix = None
        return projection_matrix

    def _apply_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        projection_matrix = self.get_proj_matrix(device=q.device, dtype=q.dtype)
        q_prime, k_prime = self.q_k_projection(q, k, projection_matrix)

        if key_padding_mask is None:
            b, h, n, d = k.shape
            key_padding_mask = torch.zeros(b, n, dtype=k.dtype, device=k.device)
        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool) # [b, 1, n, 1]

        if key_padding_mask is not None:
            k_prime.masked_fill_(key_padding_mask, 0.)

        # use full precision to perform linear attention.
        output = self._linear_attention(q_prime.float(), k_prime.float(), v.float()).to(q)
        return output

    @staticmethod
    def add_attn_specific_args(parent_parser, struct_name="attn_args", prefix=""):
        if hasattr(super(KernelizedAttention, KernelizedAttention), "add_attn_specific_args"):
            parent_parser = super(KernelizedAttention, KernelizedAttention).add_attn_specific_args(parent_parser, struct_name=struct_name, prefix=prefix)
        parser = parent_parser.add_argument_group("Attention")
        _name_prefix = prefix + "-" if len(prefix) > 1 else ""
        add_nested_argument(parser, '--{}approx-attn-dim'.format(_name_prefix), struct_name=struct_name, prefix=prefix, default=64, type=int,
                            help='number of random features')
        add_nested_argument(parser, '--{}proj-method'.format(_name_prefix), struct_name=struct_name, prefix=prefix, default='favorp', type=str,
                            help='which random feauture is used for RFA')
        add_nested_argument(parser, '--{}cos-weighting'.format(_name_prefix), struct_name=struct_name, prefix=prefix, action='store_true', default=False, help='')
        add_nested_argument(parser, '--{}sample-scheme'.format(_name_prefix), struct_name=struct_name, prefix=prefix, default='default', type=str)
        return parent_parser