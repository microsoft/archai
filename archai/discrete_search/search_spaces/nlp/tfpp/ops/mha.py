''' Modified from https://github.com/HazyResearch/flash-attention/ '''

import math
from warnings import warn
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from einops import rearrange

try:
    from flash_attn.ops.fused_dense import FusedDense
    from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding
except ImportError:
    FusedDense = None
    FlashRotaryEmbedding = None

try:
    import ft_attention
except ImportError:
    ft_attention = None

try:
    from flash_attn.modules.mha import FlashSelfAttention, _update_kv_cache
except ImportError:
    FlashSelfAttention, _update_kv_cache = None, None

from ..utils import get_optim_flag

class BaseRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base=10000, scale_base=0, device=None):
        super().__init__()
        if scale_base > 0:
            raise NotImplementedError

        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device,
                                                dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.scale_base = scale_base
        scale = ((torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim)
                 / (1.4 * dim) if scale_base > 0 else None)
        self.register_buffer("scale", scale)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
    
    def _update_cos_sin_cache(self, x, seqlen_offset=0):
        """x: (batch, seqlen, nheads, headdim) or (batch, seqlen, 3, nheads, headdim)
        """
        seqlen = x.shape[1] + seqlen_offset
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (seqlen > self._seq_len_cached or self._cos_cached.device != x.device
            or self._cos_cached.dtype != x.dtype):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=x.device, dtype=self.inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(x.dtype)
                self._sin_cached = torch.sin(freqs).to(x.dtype)
            else:
                power = ((torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device)
                          - seqlen // 2) / self.scale_base)
                scale = self.scale.to(device=power.device) ** rearrange(power, 's -> s 1')
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(x.dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(x.dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(x.dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(x.dtype)

    def apply_rotary_emb_qkv(self, qkv: torch.FloatTensor,
                             sin: torch.FloatTensor,
                             cos: torch.FloatTensor,
                             sin_k: Optional[torch.FloatTensor] = None,
                             cos_k: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        _, seqlen, three, _, headdim = qkv.shape
        assert three == 3

        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        
        assert rotary_dim <= headdim
        assert seqlen <= rotary_seqlen
        
        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        assert sin.shape == cos_k.shape == sin_k.shape == (rotary_seqlen, rotary_dim // 2)

        q_rot = qkv[:, :, 0, :, :rotary_dim]
        q_pass = qkv[:, :, 0, :, rotary_dim:]

        k_rot = qkv[:, :, 1, :, :rotary_dim]
        k_pass = qkv[:, :, 1, :, rotary_dim:]

        # Queries
        q1, q2 = q_rot.chunk(2, dim=-1)
        c, s = rearrange(cos[:seqlen], 's d -> s 1 d'), rearrange(sin[:seqlen], 's d -> s 1 d')
        
        q_rot = torch.cat([
            q1 * c - q2 * s,
            q1 * s + q2 * c
        ], axis=-1)
        
        # Keys
        k1, k2 = qkv[:, :, 1, :, :rotary_dim].chunk(2, dim=-1)
        c, s = rearrange(cos_k[:seqlen], 's d -> s 1 d'), rearrange(sin_k[:seqlen], 's d -> s 1 d')
        
        k_rot = torch.cat([
            k1 * c - k2 * s,
            k1 * s + k2 * c
        ], axis=-1)

        q = torch.cat([
            q_rot, q_pass
        ], axis=-1)

        k = torch.cat([
            k_rot, k_pass
        ], axis=-1)

        qkv = torch.cat([
            q.unsqueeze(2), k.unsqueeze(2), qkv[:, :, 2:3, :, :]
        ], axis=2)

        # inplace, but we still return it for convenience
        return qkv

    def forward(self, qkv: torch.Tensor, seqlen_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seqlen_offset: can be used in generation where the qkv being passed in is only the last
        token in the batch.
        """
        self._update_cos_sin_cache(qkv, seqlen_offset)
        return self.apply_rotary_emb_qkv(
            qkv, self._sin_cached[seqlen_offset:], self._cos_cached[seqlen_offset:]
        )


class SelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, S)
        """
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        causal = self.causal if causal is None else causal
        
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum('bthd,bshd->bhts', q, k * softmax_scale)

        if key_padding_mask is not None:
            padding_mask = torch.full((batch_size, seqlen), -10000.0, dtype=scores.dtype,
                                      device=scores.device)
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + rearrange(padding_mask, 'b s -> b 1 1 s')
        
        if causal:
            # "triu_tril_cuda_template" not implemented for 'BFloat16'
            # So we have to construct the mask in float
            causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + causal_mask.to(dtype=scores.dtype)
        
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
        output = torch.einsum('bhts,bshd->bthd', attention_drop, v)
        
        return output


class MHA(nn.Module):
    def __init__(self, hf_config: PretrainedConfig, 
                 hidden_size: int, total_heads: int, op_heads: int,
                 bias=True, dropout=0.0, softmax_scale=None, causal=True, layer_idx=None, 
                 rotary_emb_scale_base=0, return_residual=False,
                 checkpointing=False, device=None, dtype=None, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.hidden_size = hidden_size
        self.total_heads = total_heads
        self.op_heads = op_heads
        
        assert self.hidden_size % op_heads == 0, "hiddden_size must be divisible by op_heads"
        self.head_dim = self.hidden_size // total_heads
        self.op_size = op_heads * self.head_dim
        
        self.causal = causal
        self.layer_idx = layer_idx
        self.rotary_emb_dim = getattr(hf_config, 'rotary_dim', 0)
        self.fused_dense = get_optim_flag(hf_config, 'fused_dense')
        self.flash_attn = get_optim_flag(hf_config, 'flash_attn')
        self.return_residual = return_residual
        self.checkpointing = checkpointing

        if self.rotary_emb_dim > 0:
            if get_optim_flag(hf_config, 'flash_rotary_emb'):
                assert FlashRotaryEmbedding is not None, 'rotary_emb is not installed'
                self.rotary_emb = FlashRotaryEmbedding(self.rotary_emb_dim, scale_base=rotary_emb_scale_base,
                                                       device=device)
            else:
                self.rotary_emb = BaseRotaryEmbedding(self.rotary_emb_dim, scale_base=rotary_emb_scale_base,
                                                      device=device)
        else:
            warn('MHA: rotary_emb_dim is 0, no rotary embedding will be used. Performance may degrade.')

        linear_cls = nn.Linear
        if self.fused_dense:
            assert FusedDense is not None, 'Need to install fused_dense'
            linear_cls = FusedDense
        
        self.Wqkv = linear_cls(hidden_size, 3 * self.op_size, bias=bias, **factory_kwargs)
        
        if self.flash_attn:
            assert FlashSelfAttention is not None, 'flash_attn is not installed'
            self.inner_attn = FlashSelfAttention(causal=causal, softmax_scale=softmax_scale,
                                                 attention_dropout=dropout)
        else:
            self.inner_attn = SelfAttention(causal=causal, softmax_scale=softmax_scale,
                                            attention_dropout=dropout)

    def _update_kv_cache(self, kv, inference_params):
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)
        """
        assert not self.dwconv, 'Generation does not support dwconv yet'
        assert self.layer_idx is not None, 'Generation requires layer_idx in the constructor'
        return _update_kv_cache(kv, inference_params, self.layer_idx)

    def forward(self, x, x_kv=None, key_padding_mask=None, cu_seqlens=None, max_seqlen=None,
                mixer_subset=None, inference_params=None, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            x_kv: (batch, seqlen, hidden_dim), only applicable for cross-attention. If None, use x.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into x. Only applicable when using
                FlashAttention.
            max_seqlen: int. Maximum sequence length in the batch.
            key_padding_mask: boolean mask, True means to keep, False means to mask out.
                (batch, seqlen). Only applicable when not using FlashAttention.
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        if cu_seqlens is not None:
            assert max_seqlen is not None
            assert key_padding_mask is None
            assert self.flash_attn
            assert not self.dwconv
            assert self.rotary_emb_dim == 0
        
        if key_padding_mask is not None:
            assert cu_seqlens is None
            assert max_seqlen is None
            assert not self.flash_attn
        
        if inference_params is not None:
            assert key_padding_mask is None
            assert cu_seqlens is None and max_seqlen is None
            assert not self.dwconv

        attn_kwargs = ({'cu_seqlens': cu_seqlens, 'max_seqlen': max_seqlen}
                       if self.flash_attn else {'key_padding_mask': key_padding_mask})
        
        assert x_kv is None and mixer_subset is None

        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, '... (three h d) -> ... three h d', three=3, d=self.head_dim)

        if inference_params is None:
            if self.rotary_emb_dim > 0:
                qkv = self.rotary_emb(qkv)

            if not self.checkpointing:
                context = self.inner_attn(qkv, **attn_kwargs)
            else:
                context = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv, **attn_kwargs)
        else:
            if (not inference_params.fused_ft_kernel) or inference_params.sequence_len_offset == 0:
                if self.rotary_emb_dim > 0:
                    qkv = self.rotary_emb(qkv, seqlen_offset=inference_params.sequence_len_offset)
                q = qkv[:, :, 0]
                kv = self._update_kv_cache(qkv[:, :, 1:], inference_params)
                # If we're processing the prompt, causal=None (use self.causal).
                # If we're decoding, then causal=False.
                causal = None if inference_params.sequence_len_offset == 0 else False
                context = self.inner_cross_attn(q, kv, causal=causal)
            else:
                assert inference_params.fused_ft_kernel
                assert ft_attention is not None
                context = ft_attention.single_query_attention(
                    *rearrange(qkv, 'b 1 three h d -> b three h d').unbind(dim=1),
                    *inference_params.key_value_memory_dict[self.layer_idx],
                    inference_params.lengths_per_sample, inference_params.sequence_len_offset,
                    self.rotary_emb_dim
                )
                context = rearrange(context, 'b h d -> b 1 h d')
        
        out = rearrange(context, '... h d -> ... (h d)')
        
        return (out, None) if not self.return_residual else ((out, x), None)
