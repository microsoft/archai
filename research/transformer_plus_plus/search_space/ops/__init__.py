from collections import namedtuple

from .causal_self_attn import CausalSelfAttention
from .hardcoded_attn import HardcodedAttention
from .sep_conv1d import SeparableConv1d
from .sgconv import SGConv

OP = namedtuple('Operation', ['cls', 'requires_extra_config'])

OPS = {
    'causal_self_attn': OP(CausalSelfAttention, True),
    'hardcoded_attn': OP(HardcodedAttention, False),
    'sep_conv1d': OP(SeparableConv1d, True),
    #'sgconv': OP(SGConv, True)
}
