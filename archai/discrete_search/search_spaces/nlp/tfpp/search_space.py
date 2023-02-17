from typing import List, Tuple, Optional, Type, Union, Callable, Dict, Any

import torch
import numpy as np

from archai.discrete_search.search_spaces.config import (
    ArchParamTree, ArchConfig, repeat_config, ConfigSearchSpace,
    DiscreteChoice
)
from archai.discrete_search.search_spaces.nlp.tfpp.modeling_codegen import CodeGenForCausalLM, CodeGenConfig
from archai.discrete_search.search_spaces.nlp.tfpp.utils import get_attn_head_simplex
from archai.discrete_search.search_spaces.nlp.tfpp.ops import OPS


def to_tuple(x: Union[Tuple[int], int]) -> Tuple[int]:
    if not isinstance(x, (tuple, list)):
        return (x, )
    return x


class TfppSearchSpace(ConfigSearchSpace):
    def __init__(self,
                 hf_config: Optional[CodeGenConfig] = None,
                 embed_dims: Union[Tuple[int], int] = (768, ),
                 inner_dims: Union[Tuple[int], int] = (3072, ),
                 total_heads: Union[Tuple[int], int] = (12, 24),
                 total_layers: Union[Tuple[int], int] = (8, 10, 12, 16, 18),
                 local_attn_window_sizes: Union[Tuple[int], int] = (256, ),
                 sgconv_kernel_sizes: Union[Tuple[int], int] = (256, ),
                 sconv1d_kernel_sizes: Union[Tuple[int], int] = (256, ),
                 lsh_attn_num_hashes: Union[Tuple[int], int] = (4, 8),
                 lsh_attn_bucket_size: Union[Tuple[int], int] = (64,),
                 op_subset: Optional[Tuple[str]] = None,
                 mixed_ops: bool = True, 
                 homogeneous: bool = False,
                 seed: Optional[int] = None,
                 **kwargs) -> None:
        hf_config = hf_config or CodeGenConfig()
        op_subset = {
            op_name: op for op_name, op in OPS.items()
            if op_name in (op_subset or list(OPS.keys()))
        }

        if mixed_ops:
            op_allocations = get_attn_head_simplex(total_heads, list(op_subset.keys()))
        else:
            op_allocations = [
                tuple([
                    (op_name, float(item)) for op_name, item in zip(op_subset.keys(), alloc)
                ])
                for alloc in np.eye(len(OPS), dtype=np.uint).tolist()
            ]

        to_tuple = lambda x: (x, ) if not isinstance(x, (tuple, list)) else x
        arch_param_tree = ArchParamTree({
            'hidden_size': DiscreteChoice(to_tuple(embed_dims)),
            
            'hidden_layers': repeat_config({
                'total_heads': DiscreteChoice(to_tuple(total_heads)),
                'op_allocation': DiscreteChoice(op_allocations),
                'd_inner': DiscreteChoice(to_tuple(inner_dims)),

                'sgconv': {
                    'kernel_size': DiscreteChoice(to_tuple(sgconv_kernel_sizes))
                },

                'sep_conv1d': {
                    'kernel_size': DiscreteChoice(to_tuple(sconv1d_kernel_sizes))
                },

                'local_attn': {
                    'window_size': DiscreteChoice(to_tuple(local_attn_window_sizes))
                },

                'lsh_attn': {
                    'num_hashes': DiscreteChoice(to_tuple(lsh_attn_num_hashes)),
                    'bucket_size': DiscreteChoice(to_tuple(lsh_attn_bucket_size))
                }
            }, repeat_times=total_layers, share_arch=homogeneous)
        })

        super().__init__(
            CodeGenForCausalLM,
            arch_param_tree,
            model_kwargs={'hf_config': hf_config},
            seed=seed,
            **kwargs
        )
