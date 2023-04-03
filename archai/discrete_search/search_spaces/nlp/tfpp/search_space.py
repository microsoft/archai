from typing import Union, Tuple, Optional
import numpy as np
from archai.discrete_search.search_spaces.config import (
    ArchParamTree, repeat_config, ConfigSearchSpace,
    DiscreteChoice
)

from .model import LanguageModel
from .ops import OPS
from .utils import get_attn_head_simplex


def to_tuple(x: Union[Tuple[int], int]) -> Tuple[int]:
    if not isinstance(x, (tuple, list)):
        return (x, )
    return x


class TfppSearchSpace(ConfigSearchSpace):
    def __init__(self,
                 backbone: str = 'codegen',
                 embed_dims: Union[Tuple[int], int] = (768, ),
                 inner_dims: Union[Tuple[int], int] = (3072, ),
                 total_heads: Union[Tuple[int], int] = (12,),
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
                 disable_cache: bool = True,
                 **hf_config_kwargs) -> None:
        op_subset = {
            op_name: op for op_name, op in OPS.items()
            if op_name in (op_subset or list(OPS.keys())) and not op.deprecated
        }

        if disable_cache:
            hf_config_kwargs['use_cache'] = False

        if mixed_ops:
            op_allocations = get_attn_head_simplex(total_heads, list(op_subset.keys()), grid_scale=2)
        else:
            op_allocations = [
                tuple([
                    (op_name, float(item)) for op_name, item in zip(op_subset.keys(), alloc)
                ])
                for alloc in np.eye(len(op_subset), dtype=np.uint).tolist()
            ]

        to_tuple = lambda x: (x, ) if not isinstance(x, (tuple, list)) else x

        arch_param_tree = ArchParamTree({
            'backbone': backbone,

            'hidden_size': DiscreteChoice(to_tuple(embed_dims)),
            
            'hidden_layers': repeat_config({
                'total_heads': DiscreteChoice(to_tuple(total_heads)),
                'op_allocation': DiscreteChoice(op_allocations),
                'd_inner': DiscreteChoice(to_tuple(inner_dims)),

                'sgconv': {
                    'kernel_size': DiscreteChoice(to_tuple(sgconv_kernel_sizes))
                },

                'sgconv3': {
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
            LanguageModel,
            arch_param_tree,
            model_kwargs=(hf_config_kwargs or {}),
            seed=seed,
        )
