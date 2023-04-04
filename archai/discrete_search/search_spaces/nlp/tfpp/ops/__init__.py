from collections import namedtuple

from .mha import MHA
from .causal_self_attn import CausalSelfAttention
from .sep_conv1d import SeparableConv1d
from .sgconv import SGConv
from .sgconv3 import SGConv3
from .local_attention import LocalMHA
from .lsh_attn import LSHAttention

OP = namedtuple(
    'Operation', ['cls', 'requires_extra_config', 'deprecated'],
    defaults=[None, None, False]
)

OPS = {
    'causal_self_attn': OP(CausalSelfAttention, False, deprecated=True), # For retro-compatibility
    'flash_mha': OP(MHA, False, deprecated=True), # For retro-compatibility
    'mha': OP(MHA, False),
    'sep_conv1d': OP(SeparableConv1d, True),
    'sgconv': OP(SGConv, True),
    'sgconv3': OP(SGConv3, True),
    'local_attn': OP(LocalMHA, True),
    'lsh_attn': OP(LSHAttention, True)
}
