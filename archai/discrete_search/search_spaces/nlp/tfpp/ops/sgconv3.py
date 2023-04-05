# Modified from S4: https://github.com/HazyResearch/state-spaces/blob/main/src/models/sequence/ss/s4.py
from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from einops import rearrange, repeat
import opt_einsum as oe

from archai.discrete_search.search_spaces.config import ArchConfig

from ..utils import get_optim_flag
from .sgconv import GConv

optimized = True

if optimized:
    contract = oe.contract
else:
    contract = torch.einsum

try:
    from .fftconv_ import fftconv_func
except ImportError:
    fftconv_func = None


@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)

class GConv3(GConv):
    requires_length = True

    def __init__(
        self,
        d_model,
        d_state=64,
        l_max=1,  # Maximum length of sequence. Fine if not provided: the kernel will keep doubling in length until longer than sequence. However, this can be marginally slower if the true length is not a power of 2
        head_dim=1,  # maps to head dim in H3
        channels=1,  # maps 1-dim to C-dim
        bidirectional=False,
        # Arguments for FF
        activation='gelu',  # activation in between SS and FF
        ln=False,  # Extra normalization
        postact=None,  # activation after FF
        initializer=None,  # initializer on FF
        weight_norm=False,  # weight normalization on FF
        hyper_act=None,  # Use a "hypernetwork" multiplication
        use_fast_fftconv=False,
        dropout=0.0,
        transposed=True,  # axis ordering (B, L, D) or (B, D, L)
        verbose=False,
        shift=False,
        linear=False,
        mode="cat_randn",
        # SSM Kernel arguments
        **kernel_args,
    ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, set l_max=1
        channels: can be interpreted as a number of "heads"
        bidirectional: bidirectional
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, H) or (B, H, L) [B=batch size, L=sequence length, H=hidden dimension]

        Other options are all experimental and should not need to be configured
        """
        assert bidirectional == False, 'currently GConv4 does not support bidirectional=True'
        assert channels == 1, 'channels should be set to 1 for GConv3, select number of heads with the head_dim parameter'
        
        super().__init__(d_model=d_model, d_state=d_state, l_max=l_max, channels=channels, 
                            bidirectional=bidirectional, activation=activation, ln=ln, 
                            postact=postact, initializer=initializer, weight_norm=weight_norm,  
                            hyper_act=hyper_act, use_fast_fftconv=use_fast_fftconv, dropout=dropout,
                            transposed=transposed, verbose=verbose, shift=shift, linear=linear,
                            mode=mode, **kernel_args)

        self.d_model = d_model
        self.head_dim  = head_dim
        assert d_model % head_dim == 0
        self.h = d_model // head_dim
        # if self.use_fast_fftconv and not self.head_dim in [1,8]:
        #     print('fast fftconv only supported for head_dim of 1 or 8')
        #     self.use_fast_fftconv = False
        
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)

        # self.init_scale = kernel_args.get('init_scale', 0)
        # self.kernel_dim = kernel_args.get('kernel_dim', 64)
        # self.num_scales = kernel_args.get('n_scales', None)
        # if self.num_scales is None:
        #     self.num_scales = 1 + math.ceil(math.log2(l_max/self.kernel_dim)) - self.init_scale
        
        decay_min = kernel_args.get('decay_min', 2)
        decay_max = kernel_args.get('decay_max', 2)

        self.kernel_list_key = self.init_kernels(h=self.d_model, **kernel_args)
        self.D_key = nn.Parameter(torch.randn(channels, self.d_model))

        self.kernel_list = self.init_kernels(h=self.h, **kernel_args)
        self.D = nn.Parameter(torch.randn(channels, self.h))

        if 'learnable' in mode:
            self.decay_key = nn.Parameter(torch.rand(self.d_model) * (decay_max - decay_min) + decay_min)
            self.decay = nn.Parameter(torch.rand(self.h) * (decay_max - decay_min) + decay_min)
            if 'fixed' in mode:
                self.decay_key.requires_grad = False
                self.decay.requires_grad = False
            else:
                self.decay_key._optim = {'lr': kernel_args.get('lr', 0.001),}
                self.decay._optim = {'lr': kernel_args.get('lr', 0.001),}
            self.register_buffer('multiplier_key', torch.tensor(1.0))
            self.register_buffer('multiplier', torch.tensor(1.0))
        else:
            self.register_buffer('multiplier_key', torch.linspace(decay_min, decay_max, self.d_model).view(1, -1, 1))
            self.register_buffer('multiplier', torch.linspace(decay_min, decay_max, self.h).view(1, -1, 1))

        self.register_buffer('kernel_norm_key', torch.ones(channels, self.d_model, 1))
        self.register_buffer('kernel_norm_initialized_key', torch.tensor(0, dtype=torch.bool))
        self.register_buffer('kernel_norm', torch.ones(channels, self.h, 1))
        self.register_buffer('kernel_norm_initialized', torch.tensor(0, dtype=torch.bool))

        self.pw_linear = nn.Linear(self.d_model, self.d_model)

    def init_kernels(self, h, **kernel_args):
        kernel_list = nn.ParameterList()
        for _ in range(self.num_scales):
            if 'randn' in self.mode:
                kernel = nn.Parameter(torch.randn(self.channels, h, self.kernel_dim))
            elif 'cos' in self.mode:
                kernel = nn.Parameter(torch.cat([torch.cos(torch.linspace(0, 2*i*math.pi, self.kernel_dim)).expand(
                    self.channels, 1, self.kernel_dim) for i in range(h)], dim=1)[:, torch.randperm(h), :])
            else:
                raise ValueError(f"Unknown mode {self.mode}")
            kernel._optim = {'lr': kernel_args.get('lr', 0.001),}
            kernel_list.append(kernel)
        return kernel_list

    def get_kernels_forward(self, multiplier, kernel_list_init):
        kernel_list = []
        interpolate_mode = 'nearest' if 'nearest' in self.mode else 'linear'
        if 'sum' in self.mode:
            for i in range(self.num_scales):
                kernel = F.pad(
                    F.interpolate(
                        kernel_list_init[i],
                        scale_factor=2**(i + self.init_scale),
                        mode=interpolate_mode,
                    ),
                    (0, self.kernel_dim*2**(self.num_scales - 1 + self.init_scale) -
                     self.kernel_dim*2**(i + self.init_scale)),
                ) * multiplier ** (self.num_scales - i - 1)
                kernel_list.append(kernel)
            k = sum(kernel_list)
        elif 'cat' in self.mode:
            for i in range(self.num_scales):
                kernel = F.interpolate(
                    kernel_list_init[i],
                    scale_factor=2**(max(0, i-1) + self.init_scale),
                    mode=interpolate_mode,
                ) * multiplier ** (self.num_scales - i - 1)
                kernel_list.append(kernel)
            k = torch.cat(kernel_list, dim=-1)
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        return k
    
    
    # absorbs return_output and transformer src mask
    def forward(self, u, return_kernel=False):
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)
        if self.use_fast_fftconv and L % 2 != 0:
            u = F.pad(u, (0, 1))

        k_key = self.get_kernels_forward(self.multiplier_key, self.kernel_list_key)
        k = self.get_kernels_forward(self.multiplier, self.kernel_list)

        if 'learnable' in self.mode:
            k_key = k_key * torch.exp(-self.decay_key.view(1, -1, 1)*torch.log(
                torch.arange(k_key.size(-1), device=k_key.device)+1).view(1, 1, -1))
            k = k * torch.exp(-self.decay.view(1, -1, 1)*torch.log(
                torch.arange(k.size(-1), device=k.device)+1).view(1, 1, -1))

        if not self.kernel_norm_initialized:
            self.kernel_norm_key = k_key.norm(dim=-1, keepdim=True).detach()
            self.kernel_norm_initialized_key = torch.tensor(1, dtype=torch.bool, device=k.device)
            self.kernel_norm = k.norm(dim=-1, keepdim=True).detach()
            self.kernel_norm_initialized = torch.tensor(1, dtype=torch.bool, device=k.device)
            if self.verbose:
                print(f"Key Kernel norm: {self.kernel_norm_key.mean()}, Kernel norm: {self.kernel_norm.mean()}")
                print(f"Key Kernel size: {k_key.size()}, Kernel size: {k.size()}")

        k_key = k_key[..., :L] if k_key.size(-1) >= L else F.pad(k_key, (0, L - k_key.size(-1)))
        k = k[..., :L] if k.size(-1) >= L else F.pad(k, (0, L - k.size(-1)))

        k_key = k_key / self.kernel_norm_key  # * (L / self.l_max) ** 0.5
        k = k / self.kernel_norm  # * (L / self.l_max) ** 0.5

        # Convolution
        if self.bidirectional:
            raise NotImplementedError

        # compute key, query, and value
        u = rearrange(u, 'b h l -> h (b l)')  # (H B*L)
        dtype = (self.q_proj.weight.dtype if not torch.is_autocast_enabled()
                 else torch.get_autocast_gpu_dtype())
        query = self.q_proj.weight @ u + self.q_proj.bias.to(dtype).unsqueeze(-1)
        key = self.k_proj.weight @ u + self.k_proj.bias.to(dtype).unsqueeze(-1)  # (H B*L)
        value = self.v_proj.weight @ u + self.v_proj.bias.to(dtype).unsqueeze(-1)
        query, key, value = [rearrange(x, 'h (b l) -> b h l', l=L) for x in [query, key, value]]

        # first conv
        k_key = rearrange(k_key, '1 h l -> h l')
        if self.use_fast_fftconv:
            dropout_mask = None
            # No GeLU after the SSM
            # We want output_hbl=True so that k has the same layout as q and v for the next
            # fftconv
            key = fftconv_func(key, k_key, self.D_key.squeeze(0), dropout_mask, False, False, True)
            # This line below looks like it doesn't do anything, but it gets the stride right
            # for the case batch_size=1. In that case k has stride (L, L, 1), but q and v has
            # stride (H * L, L, 1). The two strides are equivalent because batch_size=1, but
            # the C++ code doesn't like that.
            key = rearrange(rearrange(key, 'b h l -> h b l'), 'h b l -> b h l')
        else:
            fft_size = 2*L 
            k_key_f = torch.fft.rfft(k_key, n=fft_size)  # (H L+1)
            key_f = torch.fft.rfft(key, n=fft_size)  # (B H L+1)
            y_f = contract('bhl,hl->bhl', key_f, k_key_f)
            y = torch.fft.irfft(y_f, n=fft_size)[..., :L]  # (B H L)
            # Compute D term in state space equation - essentially a skip connection
            key = y + contract('bhl,1h->bhl', key, self.D_key)

        # second conv
        k = rearrange(k, '1 h l -> h l') # (H L)
        if self.use_fast_fftconv:
            if self.head_dim in [1,8]:
                dropout_mask = None
                # No GeLU after the SSM
                # Set output_hbl_layout=True since we'll be doing a matmul right after
                y = fftconv_func(key, k, self.D.squeeze(0), dropout_mask, 
                                False, False, True, value, self.head_dim, query)
            else:
                kv = (rearrange(key, 'b (h d1) l -> b d1 1 h l', d1=self.head_dim)
                        * rearrange(value, 'b (h d2) l -> b 1 d2 h l', d2=self.head_dim))  # B d1 d2 h L
                kv = rearrange(kv, 'b d1 d2 h l -> b (d1 d2 h) l')
                k = repeat(k, 'h l -> d h l', d=self.head_dim**2).clone().contiguous()
                k = rearrange(k, 'd h l -> (d h) l')
                D = repeat(self.D, '1 h -> d h', d=self.head_dim**2).clone().contiguous()
                D = rearrange(D, 'd h -> (d h)')

                y = fftconv_func(kv, k, D, None, False, False, True)
                y = rearrange(y, 'b (d1 d2 h) l -> b d1 d2 h l', d1=self.head_dim, d2=self.head_dim)
                query = rearrange(query, 'b (h d1) l -> b d1 1 h l', d1=self.head_dim)
                # einsum is way slower than multiply and then sum.
                y = mul_sum(y, query)
                y = rearrange(y, 'b d h l -> b (d h) l')
        else:
            fft_size = 2*L
            kv = (rearrange(key, 'b (h d1) l -> b d1 1 h l', d1=self.head_dim)
                    * rearrange(value, 'b (h d2) l -> b 1 d2 h l', d2=self.head_dim))  # B d1 d2 h L
            kv_f = torch.fft.rfft(kv, n=fft_size) / fft_size
            k_f = torch.fft.rfft(k, n=fft_size)  # H L+1
            y = torch.fft.irfft(kv_f * k_f, n=fft_size, norm='forward')[..., :L]  # B d1 d2 h L
            y = y + kv * self.D.unsqueeze(-1)  # B d1 d2 h L
            query = rearrange(query, 'b (h d1) l -> b d1 1 h l', d1=self.head_dim)
            # einsum is way slower than multiply and then sum.
            if self.head_dim > 1:
                y = mul_sum(y, query)
                y = rearrange(y, 'b d h l -> b (d h) l')
            else:
                y = rearrange(y * query, 'b 1 1 h l -> b h l')

        # Reshape to flatten channels
        # y = rearrange(y, '... c h l -> ... (c h) l')

        if not self.linear:
            y = self.dropout(self.activation(y))

        if not self.transposed:
            y = y.transpose(-1, -2)

        if not self.linear:
            y = self.norm(y)
            y = self.output_linear(y)
        # y = self.pw_linear(y)

        if return_kernel:
            return y, k
        return y, None

    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)


class SGConv3(nn.Module):
    def __init__(self, arch_config: ArchConfig, hidden_size: int,
                 total_heads: int, op_heads: int, 
                 hf_config: PretrainedConfig, **kwargs):
        super().__init__()
        assert hidden_size % total_heads == 0

        self.hidden_size = hidden_size
        self.total_heads = total_heads
        self.op_heads = op_heads
        
        # Architecture params
        self.kernel_size = arch_config.pick('kernel_size')
        self.use_fast_fftconv = get_optim_flag(hf_config, 'fast_fftconv')
        self.channels = 1
        
        self.op_size = op_heads * (hidden_size // total_heads)
        self.in_proj = nn.Sequential(
            nn.Linear(hidden_size, self.op_size * 2),
            nn.GLU(dim=-1)
        )

        self.sgconv = GConv3(
            self.op_size, l_max=hf_config.max_position_embeddings,
            head_dim=self.op_heads, channels=self.channels, kernel_dim=self.kernel_size,
            use_fast_fftconv=self.use_fast_fftconv,
            transposed=False, verbose=False
        )

        self.act = nn.GELU(approximate='none')

    def forward(self, x: torch.Tensor, **kwargs):
        output, _ = self.sgconv(self.in_proj(x))
        return self.act(output), None


if __name__ == '__main__':
    B = 2  # batch size
    H = 768  # d_model
    L = 2048 # sequence length
    device = 'cuda'

    import torch.utils.benchmark as benchmark

    flash_layer = GConv3(d_model=H, l_max=L, head_dim=12, kernel_dim=128, use_fast_fftconv=True, transposed=False).to(device)
    layer = GConv3(d_model=H, l_max=L, head_dim=8, kernel_dim=128, use_fast_fftconv=False, transposed=False).to(device)
    u = torch.randn(B, L, H, device=device, dtype=torch.float32, requires_grad=True)

    t0 = benchmark.Timer(
            stmt='flash_layer(u)',
            globals={'flash_layer': flash_layer, 'u': u})
    t1 = benchmark.Timer(
            stmt='layer(u)',
            globals={'layer': layer, 'u': u})
    print(t0.timeit(100))
    print(t1.timeit(100))