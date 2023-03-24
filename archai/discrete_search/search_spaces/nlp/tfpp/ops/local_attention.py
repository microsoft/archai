'''Adapted from https://github.com/lucidrains/local-attention.'''

import math
from typing import Optional

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat, pack, unpack

from archai.discrete_search.search_spaces.config import ArchConfig

TOKEN_SELF_ATTN_VALUE = -5e4


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n = x.shape[-2]
        t = torch.arange(n, device = x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

def rotate_half(x):
    x = rearrange(x, 'b ... (r d) -> b (...) r d', r = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(q, k, freqs):
    q, k = map(lambda t: (t * freqs.cos()) + (rotate_half(t) * freqs.sin()), (q, k))
    return q, k

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return False, tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value = value)

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value = pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim = dim)


class LocalAttention(nn.Module):
    def __init__(
        self,
        window_size,
        causal = False,
        look_backward = 1,
        look_forward = None,
        dropout = 0.,
        autopad = False,
        exact_windowsize = False,
        pad_value: int = -1,
        rel_pos_emb_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        look_forward = look_forward or (0 if causal else 1)
        assert not (causal and look_forward > 0)

        self.window_size = window_size
        self.autopad = autopad
        self.exact_windowsize = exact_windowsize

        self.causal = causal

        self.look_backward = look_backward
        self.look_forward = look_forward
        self.pad_value = pad_value

        self.dropout = nn.Dropout(dropout)
        
        self.rel_pos = None
        
        if rel_pos_emb_dim is not None:  # backwards compatible with old `rel_pos_emb_config` deprecated argument
            self.rel_pos = SinusoidalEmbeddings(rel_pos_emb_dim)

    def forward(self, q, k, v, bin_attention_mask: Optional[torch.FloatTensor] = None):
        # https://github.com/arogozhnikov/einops/blob/master/docs/4-pack-and-unpack.ipynb
        (q, packed_shape), (k, _), (v, _) = map(lambda t: pack([t], '* n d'), (q, k, v))

        if self.rel_pos is not None:
            pos_emb = self.rel_pos(q)
            q, k = apply_rotary_pos_emb(q, k, pos_emb)

        # auto padding
        if self.autopad:
            orig_seq_len = q.shape[1]
            (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, self.window_size, dim = -2), (q, k, v))

        b, n, dim_head, device, dtype = *q.shape, q.device, q.dtype
        scale = dim_head ** -0.5

        assert (n % self.window_size) == 0, f'sequence length {n} must be divisible by window size {self.window_size} for local attention'

        windows = n // self.window_size

        seq = torch.arange(n, device = device)
        b_t = rearrange(seq, '(w n) -> 1 w n', w = windows, n = self.window_size)

        bq, bk, bv = map(lambda t: rearrange(t, 'b (w n) d -> b w n d', w = windows), (q, k, v))

        look_around_kwargs = dict(
            backward =  self.look_backward,
            forward =  self.look_forward,
            pad_value = self.pad_value
        )

        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        bq_t = rearrange(bq_t, '... i -> ... i 1')
        bq_k = rearrange(bq_k, '... j -> ... 1 j')

        sim = einsum('b h i e, b h j e -> b h i j', bq, bk) * scale

        mask_value = max_neg_value(sim)

        if self.causal:
            causal_mask = bq_t < bq_k

            if self.exact_windowsize:
                max_causal_window_size = (self.window_size * self.look_backward)
                causal_mask = causal_mask | (bq_t > (bq_k + max_causal_window_size))

            sim = sim.masked_fill(causal_mask, mask_value)
            del causal_mask

        # mask out padding value
        if self.autopad and needed_pad:
            pad_mask = bq_k == self.pad_value
            sim = sim.masked_fill(pad_mask, mask_value)
            del pad_mask

        if bin_attention_mask is not None:
            mask = bin_attention_mask.bool()
            batch = bin_attention_mask.shape[0]
            assert (b % batch) == 0

            h = b // bin_attention_mask.shape[0]

            if self.autopad:
                _, mask = pad_to_multiple(mask, self.window_size, dim=-1, value=False)

            mask = rearrange(mask, '... (w n) -> (...) w n', w = windows, n = self.window_size)
            mask = look_around(mask, **{**look_around_kwargs, 'pad_value': False})
            mask = rearrange(mask, '... j -> ... 1 j')
            mask = repeat(mask, 'b ... -> (b h) ...', h = h)
            sim = sim.masked_fill(~mask, mask_value)
            del mask

        # attention
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregation
        out = einsum('b h i j, b h j e -> b h i e', attn, bv)
        out = rearrange(out, 'b w n d -> b (w n) d')

        if self.autopad:
            out = out[:, :orig_seq_len, :]

        out, *_ = unpack(out, packed_shape, '* n d')
        return out


class LocalMHA(nn.Module):
    def __init__(
        self,
        arch_config: ArchConfig,
        hidden_size: int,
        total_heads: int,
        op_heads: int,
        att_dropout = 0.,
        prenorm = False,
        use_rotary: bool = True,
        **kwargs
    ):
        super().__init__()
        assert hidden_size % total_heads == 0, 'hidden size must be divisible by total heads'

        self.hidden_size = hidden_size
        self.total_heads = total_heads
        self.op_heads = op_heads

        head_size = self.hidden_size // self.total_heads
        self.op_size = head_size * self.op_heads

        self.norm = nn.LayerNorm(hidden_size) if prenorm else None
        self.to_qkv = nn.Linear(hidden_size, self.op_size * 3, bias = False)

        self.attn_fn = LocalAttention(
            window_size = arch_config.pick('window_size'),
            causal = True,
            autopad = True,
            exact_windowsize = True,
            dropout=att_dropout,
            rel_pos_emb_dim=(head_size if use_rotary else None),
            **kwargs
        )

    def forward(self, hidden_states, bin_attention_mask: Optional[torch.LongTensor] = None, **kwargs):
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        q, k, v = self.to_qkv(hidden_states).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.op_heads), (q, k, v)) 

        out = self.attn_fn(q, k, v, bin_attention_mask=bin_attention_mask)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return out, None
