import math
from typing import Dict, Optional, Tuple, List
import logging
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import uuid
from einops import rearrange
from efficient_attention import add_nested_argument
from efficient_attention.attn_utils import pad_to_multiple

logger = logging.getLogger(__name__)


def prm_projection(
    data: torch.Tensor,
    projection_matrix: torch.Tensor,
    normalize: bool=True
    ):
    """
    Constructs nonnegative kernel features for fast softmax attention.
    Args:
    data: input for which features are computes
    projection_matrix: random matrix used to compute features
    eps: numerical stabilizer.
    Returns:
    Random features for fast softmax attention.
    """
    # data : [b, h, lk, d]
    # proj : [b, h, lc, d]
    data_normalizer = (data.shape[-1] ** -0.5)
    data_dash = torch.einsum('...nd,...md->...nm', 
                            projection_matrix,
                            (data_normalizer * data),
                            ) # [b, h, lq, lk]
    # norm = (data_normalizer ** 2) * torch.sum(data ** 2, dim=-1).unsqueeze(-2) / 2.0# [b, h, 1, lk]
    norm = data_normalizer * torch.sum(data ** 2, dim=-1).unsqueeze(-2) / 2.0# [b, h, 1, lk]
    if normalize:
        proj_data = F.softmax(data_dash - norm, dim=-1)  # [b, h, l_c, l_k]      
    else:
        proj_data = data_dash - norm
    return proj_data


# adapted from 
# https://github.com/lucidrains/FLASH-pytorch/blob/main/flash_pytorch/flash_pytorch.py#L54
class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale,
        causal = False,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)

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
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j 1 -> i j')
        return bias * self.scale

def window_1d_merge(x):
    return rearrange(x, '... g w d ->... (g w) d')

def causal_window_1d_partition(x, window_size, ext_window_size=0, pad_val=0):
    b, h, n, d = x.shape
    n_groups = n // window_size
    if ext_window_size > 0:
        ext_len = ext_window_size
        x = F.pad(x, (0, 0, ext_len, 0), value=pad_val)
        out_shape = (b, h, n_groups, ext_len + window_size, d)
        strides = x.stride()
        out_stride = (strides[0], strides[1], window_size * strides[2], strides[2], strides[3])
        return torch.as_strided(x, size=out_shape, stride=out_stride)
    else:
        return rearrange(x, '... (g w) d -> ... g w d', w=window_size)

def default(val, d):
    return val if val is not None else d

def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        assert (
            module.weight.size(1) % block_size == 0
        ), "Input features must be a multiple of block sizes"

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            assert (
                module.in_channels % block_size == 0
            ), "Input channels must be a multiple of block sizes"
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(
                    in_features // block_size * out_features, device=weight.device
                )
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(
                        weight.size(0), weight.size(1), device=weight.device
                    )
                    mask.bernoulli_(p)
                    mask = (
                        mask.unsqueeze(2)
                        .unsqueeze(3)
                        .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                    )

            # scale weights and apply mask
            mask = mask.to(
                torch.bool
            )  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module

# copied Fairseq modules here to avoid direct dependencies
class FairseqDropout(nn.Module):
    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, inplace: bool = False):
        if self.p > 0 and (self.training or self.apply_during_inference):
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

    def make_generation_fast_(
        self,
        name: str,
        retain_dropout: bool = False,
        retain_dropout_modules: Optional[List[str]] = None,
        **kwargs
    ):
        if retain_dropout:
            if retain_dropout_modules is not None and self.module_name is None:
                logger.warning(
                    "Cannot enable dropout during inference for module {} "
                    "because module_name was not set".format(name)
                )
            elif (
                retain_dropout_modules is None  # if None, apply to all modules
                or self.module_name in retain_dropout_modules
            ):
                logger.info(
                    "Enabling dropout during inference for module: {}".format(name)
                )
                self.apply_during_inference = True
            else:
                logger.info("Disabling dropout for module: {}".format(name))

class FairseqIncrementalState(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_incremental_state()

    def init_incremental_state(self):
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key: str) -> str:
        return "{}.{}".format(self._incremental_state_id, key)

    def get_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
    ) -> Optional[Dict[str, Optional[Tensor]]]:
        """Helper for getting incremental state for an nn.Module."""
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
        value: Dict[str, Optional[Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
        """Helper for setting incremental state for an nn.Module."""
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state


def with_incremental_state(cls):
    cls.__bases__ = (FairseqIncrementalState,) + tuple(
        b for b in cls.__bases__ if b != FairseqIncrementalState
    )
    return cls

@with_incremental_state
class CausalEVAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        self_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        attn_args=None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.window_size = attn_args.window_size
        if attn_args.overlap_window:
            self.ext_size = max(1, self.window_size)
        else:
            self.ext_size = 0
        
        self.causal = attn_args.causal
        self.num_chunks = attn_args.num_chunks
        self.chunk_size = attn_args.chunk_size
        if self.chunk_size is not None:
            assert self.window_size >= self.chunk_size and self.window_size % self.chunk_size == 0
            # chunk_size overrides the number of landmarks
            self.num_chunks = None

        self.use_t5_rpe = (attn_args.use_t5_rpe) if attn_args.window_size > 0 else False

        if self.use_t5_rpe:
            self.rel_pos_bias = T5RelativePositionBias(
                self.scaling, 
                causal = self.causal, 
                num_buckets=max(min(int((self.window_size + self.ext_size) / 2), 64), 16), 
                max_distance=attn_args.window_size + self.ext_size
            )
        else:
            self.rel_pos_bias = None

        self.adaptive_proj = attn_args.adaptive_proj
        if self.adaptive_proj in ['qk']:
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
        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        if hasattr(self, "adaptive_mu_q"):
            self.adaptive_mu_q.apply(self._init_weights)

        if hasattr(self, "adaptive_mu_k"):
            self.adaptive_mu_k.apply(self._init_weights)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def _process_input(self, x, key_padding_mask):
        # this function re-implements the parent method.
        B, N, C = x.shape
        if self.window_size > 0:
            if key_padding_mask is None:
                x, key_padding_mask = pad_to_multiple(x, self.window_size, dim=-2, create_mask=True)
            else:
                x = pad_to_multiple(x, self.window_size, dim=-2)
                key_padding_mask = pad_to_multiple(key_padding_mask, self.window_size, dim=-1, value=True)
            N = x.shape[-2]
        return x, key_padding_mask


    def window_partition(self, x, shape, ext_window_size, pad_val=0, window_size=None):
        window_size = default(window_size, self.window_size)
        return causal_window_1d_partition(
            x, 
            window_size=window_size, 
            ext_window_size=ext_window_size, 
            pad_val=pad_val
            )
    
    def window_merge(self, x, shape, window_size=None):
        window_size = default(window_size, self.window_size)
        return window_1d_merge(x)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        # static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        # before_softmax: bool = False,
        # need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        mask_val = -5e4
        query = query.transpose(0, 1)
        
        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [bsz, tgt_len, embed_dim]
        if key is not None:
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            key_bsz, src_len , _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert bsz, src_len == value.shape[:2]

        if incremental_state is None:
            # pad the whole seq only when incremental_state is None.
            B, tgt_len, C = query.shape
            query, key_padding_mask = self._process_input(query, key_padding_mask)
            B, N, C = query.shape
        seq_shape = (N,)
        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q = (
            q.contiguous()
            .view(bsz, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
        else:
            saved_state = None

        if saved_state is not None:
            # saved states are stored with shape (B, H, N, D)

            # initialize chunk stats
            # only CURRENT chunk-wise statistics are needed to traced;
            # historical statistics have been stored in rf_k_bar and beta.
            chunk_query = q
            chunk_key = k
            chunk_value = v

            # initialize window stats
            # this is implemented as a sliding window (or queue)
            window_key = k
            window_value = v

            ########################################################################
            # update query, key & value.
            # Besides chunk-wise query, key & value, and window-wise key & value,
            # we also need to store rf_k_bar and beta, which computes the historical
            # per-chunk RFA and its CV respectively.
            if "rf_k_bar" in saved_state and "beta" in saved_state:
                rf_k_bar = saved_state["rf_k_bar"]
                beta = saved_state["beta"]
            else:
                rf_k_bar = None
                beta = None

            if "prev_chunk_query" in saved_state:
                prev_chunk_query = saved_state["prev_chunk_query"]
                if prev_chunk_query is not None:
                    chunk_query = torch.cat([prev_chunk_query, q], dim=-2)
            if "prev_chunk_key" in saved_state:
                prev_chunk_key = saved_state["prev_chunk_key"]
                if prev_chunk_key is not None:
                    chunk_key = torch.cat([prev_chunk_key, k], dim=-2)
            if "prev_chunk_value" in saved_state:
                prev_chunk_value = saved_state["prev_chunk_value"]
                if prev_chunk_value is not None:
                    chunk_value = torch.cat([prev_chunk_value, v], dim=-2)
                
            # dump the chunk after the len of current chunk reaches chunk_size.
            if chunk_query.shape[-2] == self.chunk_size:
                cur_rf_q_bar = self.adaptive_mu_q(chunk_query.mean(dim=-2, keepdim=True))
                cur_rf_k_bar = self.adaptive_mu_k(chunk_key.mean(dim=-2, keepdim=True))

                ########################################################
                # feel free to modify these lines of code
                # to test mu parameterization / sampling noise / etc...
                mu = cur_rf_q_bar + cur_rf_k_bar

                if self.training:
                    weights = mu + torch.randn_like(mu)
                else:
                    weights = mu    
                ########################################################

                # [b, h, j, d], [b, h, 1, d] -> [b, h, 1, j]
                log_proj_k = prm_projection(chunk_key, weights, normalize=False)

                # [b, h, 1, j] [b, h, j, d] -> [b, h, 1, d]
                cur_beta = torch.einsum('...nj,...jd->...nd', torch.softmax(log_proj_k, dim=-1), chunk_value)

                if rf_k_bar is not None and beta is not None:
                    rf_k_bar = torch.cat([rf_k_bar, cur_rf_k_bar], dim=-2)
                    beta = torch.cat([beta, cur_beta], dim=-2)
                else:
                    rf_k_bar = cur_rf_k_bar
                    beta = cur_beta
                chunk_query = None
                chunk_key = None
                chunk_value = None

            if "prev_window_key" in saved_state:
                prev_window_key = saved_state["prev_window_key"]
                assert prev_window_key is not None
                if prev_window_key.shape[-2] == self.window_size:
                    window_key = torch.cat([prev_window_key[..., 1:, :], k], dim=-2)
                else:
                    window_key = torch.cat([prev_window_key, k], dim=-2)
            if "prev_window_value" in saved_state:
                prev_window_value = saved_state["prev_window_value"]
                assert prev_window_value is not None
                if prev_window_value.shape[-2] == self.window_size:
                    window_value = torch.cat([prev_window_value[..., 1:, :], v], dim=-2)
                else:
                    window_value = torch.cat([prev_window_value, v], dim=-2)
            # NOTE: If decoding results are not good, might switch to block-wise local attention
            saved_state["rf_k_bar"] = rf_k_bar
            saved_state["beta"] = beta
            saved_state["prev_chunk_query"] = chunk_query
            saved_state["prev_chunk_key"] = chunk_key
            saved_state["prev_chunk_value"] = chunk_value
            saved_state["prev_window_key"] = window_key
            saved_state["prev_window_value"] = window_value
            
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

            # compute approx. expectation of CVs.
            # [b, h, 1, c]
            if rf_k_bar is not None:
                approx_expected_cv = torch.einsum('...nd,...cd->...nc', q, self.scaling * rf_k_bar)

            log_qk_local_dot = torch.einsum('bhie,bhje->bhij', q, window_key) * self.scaling # [b, h, i, j]
            if self.use_t5_rpe:
                log_qk_local_dot = log_qk_local_dot + self.rel_pos_bias(log_qk_local_dot)
            
            if rf_k_bar is not None:
                local_len = log_qk_local_dot.shape[-1]
                num_rfa_chunks = approx_expected_cv.shape[-1]
                attn = torch.softmax(torch.cat([log_qk_local_dot, approx_expected_cv], dim=-1), dim=-1)
                local_attn, ra_attn = torch.split(attn, [local_len, num_rfa_chunks], dim=-1)
                output_local = torch.einsum('bhij,bhjd->bhid', local_attn, window_value)
                output_snis = torch.einsum('bhic,bhcd->bhid', ra_attn, beta) 
                ######################## Combine them together ############################
                output = output_snis + output_local # [b, h, i, d]
            else:
                # compute attention weights along with normalizing constant.
                attn = torch.softmax(log_qk_local_dot, dim=-1)
                output = torch.einsum('bhij,bhjd->bhid', attn, window_value)
            ######################## Combine them together ############################
            x = output.permute(0, 2, 1, 3).reshape((B, -1, C))
            x = self.out_proj(x)
            return x.transpose(0, 1).contiguous(), None
        else:
            # Training & evaluation only. No incremental state is used.
            if key_padding_mask is None:
                key_padding_mask = torch.zeros(B, N, dtype=k.dtype, device=k.device)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool) # [b, 1, n, 1]
        
            w_q = self.window_partition(q, seq_shape, ext_window_size=0) # [b, h, w, i, d]
            w_k = self.window_partition(k, seq_shape, ext_window_size=self.ext_size)
            w_v = self.window_partition(v, seq_shape, ext_window_size=self.ext_size) # [b, h, w, j, d]

            if self.chunk_size is not None:
                rf_chunk_size = self.chunk_size
            else:
                rf_chunk_size = int(N // self.num_chunks)
            if rf_chunk_size >= N:
                rf_w_q = q
                rf_w_k = k
                rf_w_v = v
            else:
                # [b, h, c, j, d]
                rf_w_q = self.window_partition(q, seq_shape, window_size=rf_chunk_size, ext_window_size=0)
                # [b, h, c, j, d]
                rf_w_k = self.window_partition(k, seq_shape, window_size=rf_chunk_size, ext_window_size=0)
                # [b, h, c, j, d]
                rf_w_v = self.window_partition(v, seq_shape, window_size=rf_chunk_size, ext_window_size=0)
                # compute local attention
                # [b, 1, c, j, 1]
                rf_w_mask = self.window_partition(
                    key_padding_mask, 
                    seq_shape, 
                    window_size=rf_chunk_size,
                    ext_window_size=0,
                    pad_val=1
                    ).to(torch.bool)
                # print(rf_w_mask)
                rf_w_q = rf_w_q.masked_fill(rf_w_mask, 0.)
                rf_w_k = rf_w_k.masked_fill(rf_w_mask, 0.)
                rf_w_v = rf_w_v.masked_fill(rf_w_mask, 0.)

            rf_q_bar = self.adaptive_mu_q(rf_w_q.mean(dim=-2))
            rf_k_bar = self.adaptive_mu_k(rf_w_k.mean(dim=-2))
            # [b, h, c, d]
            mu = rf_q_bar + rf_k_bar
            
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
            approx_expected_cv = torch.einsum('...wid,...cd->...wic', w_q, self.scaling * rf_k_bar)
            if self.causal:
                # [b, h, j, c, c]
                b, h, j, c = q.shape[0], q.shape[1], rf_w_k.shape[-2], rf_w_k.shape[-3]
                if self.adaptive_proj in ['no-ln', 'qk']:
                    causal_mask = torch.ones(b, h, j, c, c, dtype=q.dtype, device=q.device).triu(0).transpose(-2, -3) # [b, h, c, j, c]
                    # NOTE: .triu(0) is used to remove the context of the current chunk from localized RFA.
                    # since we compute `rf_q_bar` for each chunk for random features, 
                    # it requires the future information if we compute it on the current chunk.
                    # however, note that the current chunk's information is still retained through
                    # the local attention module.
                else:
                    raise NotImplementedError("Other adaptive projection methods are not implemented yet.")
                causal_mask = self.window_merge(causal_mask, seq_shape) # [b, h, n, c]
                causal_mask = self.window_partition(causal_mask, seq_shape, ext_window_size=0).to(torch.bool) # [b, h, w, i, c]
                approx_expected_cv = approx_expected_cv.masked_fill(causal_mask, mask_val)

            # compute local attention
            mask_q = self.window_partition(
                key_padding_mask, 
                seq_shape, 
                ext_window_size=0,
                pad_val=1
                ).to(torch.bool) # [b, 1, w, i, 1]
            mask_k = self.window_partition(
                key_padding_mask, 
                seq_shape, 
                ext_window_size=self.ext_size,
                pad_val=1
                ).to(torch.bool).transpose(-1, -2) # [b, 1, w, 1, j] 
            local_dots_mask = torch.logical_or(mask_q, mask_k)
            log_qk_local_dot = torch.einsum('bhwie,bhwje->bhwij', w_q, w_k) * self.scaling # [b, h, w, i, j]
            # if self.use_headed_t5_rpe:
                # here the t5-rpe-bias has already been scaled by \sqrt{d}
                # log_qk_local_dot = log_qk_local_dot + self.headed_rel_pos_bias(log_qk_local_dot)
            if self.use_t5_rpe:
                # here the t5-rpe-bias has already been scaled by \sqrt{d}
                log_qk_local_dot = log_qk_local_dot + self.rel_pos_bias(log_qk_local_dot)

            log_qk_local_dot = log_qk_local_dot.masked_fill(local_dots_mask, mask_val)

            if self.causal:
                # e.g., if window_size = 3 and ext_size = 3, then it creates a causal_mask as follows:
                # [0 0 0 0 1 1]
                # [0 0 0 0 0 1]
                # [0 0 0 0 0 0]
                causal_mask = torch.ones_like(log_qk_local_dot).triu(1 + self.ext_size).to(torch.bool)
                log_qk_local_dot = log_qk_local_dot.masked_fill(causal_mask, mask_val)

            local_len = log_qk_local_dot.shape[-1]
            num_rfa_chunks = approx_expected_cv.shape[-1]

            # compute attention weights along with normalizing constant.
            attn = torch.softmax(torch.cat([log_qk_local_dot, approx_expected_cv], dim=-1), dim=-1)
            attn = self.dropout_module(attn)
            local_attn, ra_attn = torch.split(attn, [local_len, num_rfa_chunks], dim=-1)
            output_local = torch.einsum('bhwij,bhwjd->bhwid', local_attn, w_v)
            output_snis = torch.einsum('bhwic,bhcd->bhwid', ra_attn, beta) 
            ######################## Combine them together ############################
            output = self.window_merge(output_snis + output_local, seq_shape) # [b, h, n, d]
            x = output.permute(0, 2, 1, 3).reshape((B,) + tuple(seq_shape) + (C,))
            x = self.out_proj(x)
            if tgt_len is not None and tgt_len != N:
                x = x[..., :tgt_len, :]
            return x.transpose(0, 1).contiguous(), None


    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - prev_key_padding_mask.size(1)),
                    device=prev_key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [prev_key_padding_mask.float(), filler.float()], dim=1
                )
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - key_padding_mask.size(1)),
                    device=key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [filler.float(), key_padding_mask.float()], dim=1
                )
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value

    @staticmethod
    def add_attn_specific_args(parent_parser, struct_name="attn_args", prefix=""):
        parser = parent_parser.add_argument_group("attention")
        # add_nested_argument(parser, '--rfa-method', default='lara', type=str)
        _name_prefix = prefix + "-" if len(prefix) > 1 else ""
        add_nested_argument(parser, '--{}adaptive-proj'.format(_name_prefix), struct_name=struct_name, prefix=prefix, default='default', type=str)
        add_nested_argument(parser, '--{}num-chunks'.format(_name_prefix), struct_name=struct_name, prefix=prefix, default=None, type=int)
        add_nested_argument(parser, '--{}chunk-size'.format(_name_prefix), struct_name=struct_name, prefix=prefix, default=None, type=int)
        add_nested_argument(parser, '--{}causal'.format(_name_prefix), action='store_true', struct_name=struct_name, prefix=prefix, default=False)
        add_nested_argument(parser, '--{}use-t5-rpe'.format(_name_prefix), action='store_true', struct_name=struct_name, prefix=prefix, default=False)
        add_nested_argument(parser, '--{}window-size'.format(_name_prefix), struct_name=struct_name, prefix=prefix, default=4, type=int)
        add_nested_argument(parser, '--{}overlap-window'.format(_name_prefix), action='store_true', struct_name=struct_name, prefix=prefix, default=False)
        return parent_parser

if __name__ == '__main__':
    # a test case
    import numpy
    import random
    from argparse import Namespace

    def seed():
        torch.manual_seed(0)
        numpy.random.seed(0)
        random.seed(0)

    seed()
    attn_args = Namespace(
        adaptive_proj="qk",
        num_chunks=None,
        chunk_size=16,
        causal=True,
        use_t5_rpe=True,
        window_size=64,
        overlap_window=False,
    )
    attn = CausalEVAttention(embed_dim=128, num_heads=8, attn_args= attn_args)
    attn.eval()
    input_ids = torch.randn((512, 4, 128))
    out, attn_weight = attn(input_ids, input_ids, input_ids)
    out = out.transpose(0, 1)
    j = 25
    for i in range(26, 100):
        z = out[:, j, :]

        slices = input_ids[:, :i, :]
        x, _ = attn(slices, slices, slices)
        x = x.transpose(0, 1)
        x = x[:, j, :]
        print(i, (x - z).abs().sum())