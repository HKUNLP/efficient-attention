import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

def _fp32_softmax(x, dim):
    y = torch.softmax(x, dim=dim, dtype=torch.float32).type_as(x)
    return y

# adapted from https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py
def pad_to_multiple(tensor, multiple, dim=-2, value=0, create_mask=False):
    assert dim < 0 # only accept ``dim'' index in a reverse manner
    seqlen = int(tensor.shape[dim])
    m = seqlen / multiple
    if m.is_integer():
        if create_mask:
            return tensor, torch.zeros(size=(tensor.shape[0], tensor.shape[-2]), dtype=torch.bool, device=tensor.device)
        else:
            return tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    padded_res = F.pad(tensor, (*pad_offset, 0, remainder), value=value)
    if create_mask:
        # assume dim 0 is the batch size
        padding_mask = torch.zeros(size=(padded_res.shape[0], padded_res.shape[-2]), dtype=torch.bool, device=padded_res.device)
        padding_mask[:, -remainder:] = True
        return padded_res, padding_mask
    else:
        return padded_res


def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = -2):
    dims = (-dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value= pad_value)
    if dim == -2:
        t = x.shape[-3]
        tensors = [padded_x[..., ind:(ind + t), :, :] for ind in range(forward + backward + 1)]
    if dim == -1:
        t = x.shape[-2]
        tensors = [padded_x[..., ind:(ind + t), :] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=dim)

def log_add_exp(tensor, other, mask=None, eps=1e-5):
    if mask is None:
        mask = (1., 1.)
    else:
        assert isinstance(mask, tuple) and len(mask) == 2, 'Mask must be a tuple of size 2.'
    
    a = torch.maximum(tensor, other)
    return a + ((tensor - a).exp()*mask[0] + (other - a).exp()*mask[1] + eps).log()

def log_matmul_exp(value1, value2, eps=1e-6):
    """Numerically stable implementation of the operation

    value.exp().matmul(other.exp()).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    # val1: [..., m, c]
    # val2: [..., c, n]
    m1 = torch.amax(value1, dim=-1, keepdim=True).detach()
    m2 = torch.amax(value2, dim=-2, keepdim=True).detach()
    new_value_1 = value1 - m1
    new_value_2 = value2 - m2
    return m1 + m2 + torch.log(torch.matmul(torch.exp(new_value_1), torch.exp(new_value_2)) + eps)

# copied from https://github.com/lucidrains/logavgexp-torch/
def log_avg_exp(
    t,
    mask = None,
    dim = -1,
    eps = 1e-6,
    keepdim = False
):
    if mask is not None:
        mask_value = -torch.finfo(t.dtype).max
        t = t.masked_fill(~mask, mask_value)
        n = mask.sum(dim = dim)
        norm = torch.log(n)
    else:
        n = t.shape[dim]
        norm = math.log(n)

    t = t
    max_t = t.amax(dim = dim).detach()
    t_exp = (t - max_t.unsqueeze(dim)).exp()
    avg_exp = t_exp.sum(dim = dim).clamp(min = eps) / n
    out = torch.log(avg_exp + eps) + max_t - norm
    out = out.unsqueeze(dim) if keepdim else out
    return out

class FlattenTranspose(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.flatten(2).permute(0, 2, 1)

class SplitHeads(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
    def forward(self, x):
        b, d, H, W = x.shape
        return x.reshape(b // self.num_heads, self.num_heads * d, H * W)

class MergeHeads(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
    def forward(self, x):
        b, D, n = x.shape
        return x.reshape(b, self.num_heads, D // self.num_heads, n).transpose(-1, -2)
 


class Merger(nn.Module):
    def __init__(self, config, head_dim):
        # config is of format '<act>-<pooler>'
        super().__init__()
        self.act_fn, self.pooling_fn = config.split('-')
        if self.act_fn == 'relu':
            self.activation = nn.ReLU()
        elif self.act_fn == 'identity':
            self.activation = nn.Identity()
        elif self.act_fn == 'deepset':
            self.activation = nn.Sequential(
                nn.Linear(head_dim, head_dim),
                nn.ReLU()
            )
        else:
            raise ValueError('Unsupported activation: %s' % self.act_fn)
    
    def forward(self, x, dim=-2, keepdim=False):
        x_prime = self.activation(x)
        if self.pooling_fn == 'mean':
            x = torch.mean(x_prime, dim=dim, keepdim=keepdim)
        elif self.pooling_fn == 'max':
            x = torch.amax(x_prime, dim=dim, keepdim=keepdim)
        else:
            raise ValueError('Unsupported pooling: %s' % self.pooling_fn)
        return x


def take_along_dim(input, index, dim):
    shape = input.shape
    trailing = shape[dim + 1:]
    heading = len(shape) + dim
    # print(input.shape, index.shape, dim, trailing, heading)
    index = index.reshape(index.shape + (1,) * (-dim -1)).repeat((1,) * (heading + 1) + tuple(trailing))
    return torch.gather(input, dim=dim, index=index)

def nonoverlap_window_1d_partition(x, window_size):
    return rearrange(x, '... (g w) d -> ... g w d', w=window_size)

def window_1d_partition(x, window_size, ext_window_size=0, pad_val=0):
    b, h, n, d = x.shape
    n_groups = n // window_size
    if ext_window_size > 0:
        ext_len = ext_window_size
        x = F.pad(x, (0, 0, ext_len, ext_len), value=pad_val)
        out_shape = (b, h, n_groups, 2 * ext_len + window_size, d)
        strides = x.stride()
        out_stride = (strides[0], strides[1], window_size * strides[2], strides[2], strides[3])
        return torch.as_strided(x, size=out_shape, stride=out_stride)
    else:
        return nonoverlap_window_1d_partition(x, window_size)

def window_1d_merge(x):
    return rearrange(x, '... g w d ->... (g w) d')

# adapted from https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
def nonoverlap_window_2d_partition(x, window_size):
    """
    Args:
        x: (b, h, H, W, d)
        window_size (int): window size
    Returns:
        windows: (num_windows * num_windows, window_size * window_size, C)
    """
    *_, H, W, d = x.shape
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    output = rearrange(
        x, 
        '... (h1 h) (w1 w) d -> ... (h1 w1) (h w) d', 
        h1=num_windows_h, w1=num_windows_w, h=window_size, w=window_size
        )
    return output

def window_2d_partition(x, window_size, ext_window_size=0, pad_val=0):
    """
    Args:
        x: (b, h, H, W, d)
        window_size (int): Window size
    Returns:
        x: (b, h, num_groups, group_size, d)
    """
    if ext_window_size > 0:
        b, h, H, W, d = x.shape
        total_window_size = 2 * ext_window_size + window_size
        x = F.pad(x, [0, 0, ext_window_size, ext_window_size, ext_window_size, ext_window_size], value=pad_val)
        out_shape = [b, h, H // window_size, W // window_size,
                        total_window_size, total_window_size, d]
        in_stride = x.stride()
        out_stride = [in_stride[0], in_stride[1], in_stride[2] * window_size, in_stride[3] * window_size,
                        in_stride[2], in_stride[3], in_stride[4]]
        output = x.as_strided(size=out_shape, stride=out_stride)
        return rearrange(output, '... h1 w1 h w d -> ... (h1 w1) (h w) d')
    else:
        return nonoverlap_window_2d_partition(x, window_size)

def window_2d_merge(x, window_size, hw_tuple):
    """
    Args:
        x: (b, h, num_windows * num_windows, window_size * window_size, d)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (b, h, H, W, d)
    """
    assert isinstance(hw_tuple, (list, tuple))
    H, W = hw_tuple
    b, h, num_windows_sq, window_size_sq, d = x.shape
    assert window_size ** 2 == window_size_sq
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    assert num_windows_sq == num_windows_h * num_windows_w
    output = rearrange(
        x, 
        '... (h1 w1) (h w) d -> ... (h1 h) (w1 w) d', 
        h1=num_windows_h, w1=num_windows_w, h=window_size, w=window_size
        )
    return output


def hyperm_projection(
    data: torch.Tensor,
    projection_matrix: torch.Tensor,
    normalize: bool=False,
    diagonal: bool=False,
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
    # data : [n, b, h, lk, d]
    # proj : [n, b, h, lc, d]
    # We have e^{qk^T/sqrt{d}} = e^{q_norm k_norm^T}, where
    # w_norm = w * data_normalizer for w in {q,k}.
    # NOTE: self.scale with 0.5 could considerably stablizes training.
    # now test norm also uses scaled data: that is, multiply by data.shape[-1] ** -1.
    # normalized_data = (data.shape[-1] ** -0.5) * data
    # data_dash = torch.einsum('...nd,...md->...nm', 
    #                         projection_matrix,
    #                         normalized_data,
    #                         ) # [n, b, h, c, lq]
    # norm = torch.sum(normalized_data ** 2, dim=-1).unsqueeze(-2) / 2.0# [n, b, h, 1, lk]
    data_normalizer = (data.shape[-1] ** -0.5)
    if diagonal:
        data_dash = torch.einsum('...nd,...nd->...n', 
                            projection_matrix,
                            (data_normalizer * data),
                            ) # [n, b, h, lq, lk]
        norm = data_normalizer * torch.sum(data ** 2, dim=-1) / 2.0# [n, b, h, 1, lk]
    else:
        data_dash = torch.einsum('...nd,...md->...nm', 
                                projection_matrix,
                                (data_normalizer * data),
                                ) # [n, b, h, lq, lk]
        norm = data_normalizer * torch.sum(data ** 2, dim=-1).unsqueeze(-2) / 2.0# [n, b, h, 1, lk]
    
    proj_data = math.sqrt(1 / 2.) * (
        torch.cat([
            torch.exp(data_dash - norm -
                torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()),
            torch.exp(-data_dash - norm -
                torch.amax(-data_dash, dim=(-1, -2), keepdim=True).detach()),                  
        ], dim=-2) + eps)
    return proj_data # [b, h, c, n, 2]



def prm_projection(
    data: torch.Tensor,
    projection_matrix: torch.Tensor,
    normalize: bool=True,
    diagonal: bool=False,
    return_exp: bool=False,
    is_query: bool=False,
    eps: float=1e-8):
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
    # data : [n, b, h, lk, d]
    # proj : [n, b, h, lc, d]
    # We have e^{qk^T/sqrt{d}} = e^{q_norm k_norm^T}, where
    # w_norm = w * data_normalizer for w in {q,k}.
    # NOTE: scaler with 0.5 could considerably stablizes training.
    # now test norm also uses scaled data: that is, multiply by data.shape[-1] ** -1.
    # normalized_data = (data.shape[-1] ** -0.5) * data
    # data_dash = torch.einsum('...nd,...md->...nm', 
    #                         projection_matrix,
    #                         normalized_data,
    #                         ) # [n, b, h, c, lq]
    # norm = torch.sum(normalized_data ** 2, dim=-1).unsqueeze(-2) / 2.0# [n, b, h, 1, lk]
    data_normalizer = (data.shape[-1] ** -0.5)
    if diagonal:
        data_dash = torch.einsum('...nd,...nd->...n', 
                            projection_matrix,
                            (data_normalizer * data),
                            ) # [n, b, h, lq, lk]
        norm = data_normalizer * torch.sum(data ** 2, dim=-1) / 2.0# [n, b, h, 1, lk]
    else:
        data_dash = torch.einsum('...nd,...md->...nm', 
                                projection_matrix,
                                (data_normalizer * data),
                                ) # [n, b, h, lq, lk]
        norm = data_normalizer * torch.sum(data ** 2, dim=-1).unsqueeze(-2) / 2.0# [n, b, h, 1, lk]
    if normalize:
        proj_data = F.softmax(data_dash - norm, dim=-1)  # [n, b, h, l_c, l_k]
    elif return_exp:
        if is_query:
            proj_data = torch.exp(
                data_dash - norm - torch.amax(data_dash, dim=-2, keepdim=True).detach()) + eps       
        else:
            proj_data = torch.exp(
                data_dash - norm - torch.amax(data_dash, dim=(-1, -2, -3), keepdim=True).detach()) + eps           
    else:
        proj_data = data_dash - norm
    return proj_data

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

def buffered_future_mask(self, tensor):
    dim = tensor.size(0)
    # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
    if (
        self._future_mask.size(0) == 0
        or (not self._future_mask.device == tensor.device)
        or self._future_mask.size(0) < dim
    ):
        self._future_mask = torch.triu(
            fill_with_neg_inf(torch.zeros([dim, dim])), 1
        )
    self._future_mask = self._future_mask.to(tensor)
    return self._future_mask[:dim, :dim]