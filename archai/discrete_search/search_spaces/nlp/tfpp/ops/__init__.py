from collections import namedtuple

from archai.discrete_search.search_spaces.nlp.tfpp.ops.causal_self_attn import CausalSelfAttention
from archai.discrete_search.search_spaces.nlp.tfpp.ops.sep_conv1d import SeparableConv1d
from archai.discrete_search.search_spaces.nlp.tfpp.ops.sgconv import SGConv
from archai.discrete_search.search_spaces.nlp.tfpp.ops.local_attention import LocalMHA
from archai.discrete_search.search_spaces.nlp.tfpp.ops.lsh_attn import LSHAttention

OP = namedtuple('Operation', ['cls', 'requires_extra_config'])

OPS = {
    'causal_self_attn': OP(CausalSelfAttention, False),
    'sep_conv1d': OP(SeparableConv1d, True),
    'sgconv': OP(SGConv, True),
    'local_attn': OP(LocalMHA, True),
    'lsh_attn': OP(LSHAttention, True)
}
