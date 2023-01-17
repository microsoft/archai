from typing import List, Tuple, Optional
import numpy as np
from archai.discrete_search.search_spaces.config import (
    ArchParamTree, ArchConfig, repeat_config, ConfigSearchSpace,
    DiscreteChoice
)

from transformer_plus_plus.search_space.modeling_gpt2 import GPT2LMHeadModel, GPT2Config
from transformer_plus_plus.search_space.ops import OPS
from transformer_plus_plus.search_space.utils import get_attn_head_simplex


def build_single_op_ss(embed_dims: Tuple[int, ...] = (768,),
                       d_inners: Tuple[int, ...] = (768*4,),
                       min_layers: int = 12,
                       max_layers: int = 12,
                       total_heads: Tuple[int, ...] = (12,),
                       attn_window_props: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
                       sgconv_channels: Tuple[int, ...] = (1,),
                       sgconv_kernel_sizes: Tuple[int, ...] = (32, 64, 128, 256, 512),
                       conv1d_kernel_sizes: Tuple[int, ...] = (32, 64, 128, 256, 512),
                       hf_config: Optional[GPT2Config] = None,
                       seed: int = 1) -> ConfigSearchSpace:
    hf_config = hf_config or GPT2Config()

    # Lists all possible allocations with only one op per layer
    op_allocations = [
        tuple([
            (op_name, float(item)) for op_name, item in zip(OPS.keys(), alloc)
        ])
        for alloc in np.eye(len(OPS), dtype=np.uint).tolist()
    ]

    single_op_ss = ArchParamTree({
        'hidden_size': DiscreteChoice(embed_dims),
        'hidden_layers': repeat_config({
            'total_heads': DiscreteChoice(total_heads),
            'op_allocation': DiscreteChoice(op_allocations),
            'd_inner': DiscreteChoice(d_inners),

            'causal_self_attn': {
                'attn_window_prop': DiscreteChoice(attn_window_props)
            },

            'sgconv': {
                'channels': DiscreteChoice(sgconv_channels),
                'kernel_size': DiscreteChoice(sgconv_kernel_sizes)
            },

            'sep_conv1d': {
                'kernel_size': DiscreteChoice(conv1d_kernel_sizes)
            }
        }, repeat_times=list(range(min_layers, max_layers + 1)), share_arch=True)
    })

    return ConfigSearchSpace(
        GPT2LMHeadModel,
        single_op_ss,
        hf_config=hf_config,
        model_creation_attempts=20,
        seed=seed
    )


def build_single_op_per_layer_ss(embed_dims: Tuple[int, ...] = (768,),
                                 d_inners: Tuple[int, ...] = (768*4,),
                                 min_layers: int = 12,
                                 max_layers: int = 12,
                                 total_heads: Tuple[int, ...] = (12,),
                                 attn_window_props: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
                                 sgconv_channels: Tuple[int, ...] = (1,),
                                 sgconv_kernel_sizes: Tuple[int, ...] = (32, 64, 128, 256, 512),
                                 conv1d_kernel_sizes: Tuple[int, ...] = (32, 64, 128, 256, 512),
                                 hf_config: Optional[GPT2Config] = None,
                                 seed: int = 1) -> ConfigSearchSpace:
    hf_config = hf_config or GPT2Config()

    # Lists all possible allocations with only one op per layer
    op_allocations = [
        tuple([
            (op_name, float(item)) for op_name, item in zip(OPS.keys(), alloc)
        ])
        for alloc in np.eye(len(OPS), dtype=np.uint).tolist()
    ]

    single_op_ss = ArchParamTree({
        'hidden_size': DiscreteChoice(embed_dims),
        'hidden_layers': repeat_config({
            'total_heads': DiscreteChoice(total_heads),
            'op_allocation': DiscreteChoice(op_allocations),
            'd_inner': DiscreteChoice(d_inners),

            'causal_self_attn': {
                'attn_window_prop': DiscreteChoice(attn_window_props)
            },

            'sgconv': {
                'channels': DiscreteChoice(sgconv_channels),
                'kernel_size': DiscreteChoice(sgconv_kernel_sizes)
            },

            'sep_conv1d': {
                'kernel_size': DiscreteChoice(conv1d_kernel_sizes)
            }
        }, repeat_times=list(range(min_layers, max_layers + 1)), share_arch=False)
    })

    return ConfigSearchSpace(
        GPT2LMHeadModel,
        single_op_ss,
        hf_config=hf_config,
        model_creation_attempts=20,
        seed=seed
    )


def build_mixed_attention_hom_ss(embed_dims: Tuple[int, ...] = (768,),
                                 d_inners: Tuple[int, ...] = (768*4,),
                                 min_layers: int = 12,
                                 max_layers: int = 12,
                                 total_heads: Tuple[int, ...] = (12,),
                                 attn_window_props: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
                                 sgconv_channels: Tuple[int, ...] = (1,),
                                 sgconv_kernel_sizes: Tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512),
                                 conv1d_kernel_sizes: Tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512),
                                 hf_config: Optional[GPT2Config] = None,
                                 seed: int = 1) -> ConfigSearchSpace:
    hf_config = hf_config or GPT2Config()

    single_op_ss = ArchParamTree({
        'hidden_size': DiscreteChoice(embed_dims),
        'hidden_layers': repeat_config({
            'total_heads': DiscreteChoice(total_heads),
            'op_allocation': DiscreteChoice(get_attn_head_simplex(total_heads, list(OPS.keys()))),
            'd_inner': DiscreteChoice(d_inners),
            
            'causal_self_attn': {
                'attn_window_prop': DiscreteChoice(attn_window_props)
            },

            'sgconv': {
                'channels': DiscreteChoice(sgconv_channels),
                'kernel_size': DiscreteChoice(sgconv_kernel_sizes)
            },
            
            'sep_conv1d': {
                'kernel_size': DiscreteChoice(conv1d_kernel_sizes)
            }
        }, repeat_times=list(range(min_layers, max_layers + 1)), share_arch=True)
    })

    return ConfigSearchSpace(
        GPT2LMHeadModel,
        single_op_ss,
        hf_config=hf_config,
        model_creation_attempts=20,
        seed=seed
    )


def build_mixed_attention_het_ss(embed_dims: Tuple[int, ...] = (768,),
                                 d_inners: Tuple[int, ...] = (768*4,),
                                 min_layers: int = 1,
                                 max_layers: int = 12,
                                 total_heads: Tuple[int, ...] = (12,),
                                 attn_window_props: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
                                 sgconv_channels: Tuple[int, ...] = (1,),
                                 sgconv_kernel_sizes: Tuple[int, ...] = (32, 64, 128, 256, 512),
                                 conv1d_kernel_sizes: Tuple[int, ...] = (32, 64, 128, 256, 512),
                                 hf_config: Optional[GPT2Config] = None,
                                 seed: int = 1) -> ConfigSearchSpace:
    hf_config = hf_config or GPT2Config()

    single_op_ss = ArchParamTree({
        'hidden_size': DiscreteChoice(embed_dims),
        'hidden_layers': repeat_config({
            'total_heads': DiscreteChoice(total_heads),
            'op_allocation': DiscreteChoice(get_attn_head_simplex(total_heads, list(OPS.keys()))),
            'd_inner': DiscreteChoice(d_inners),
            
            'causal_self_attn': {
                'attn_window_prop': DiscreteChoice(attn_window_props)
            },

            'sgconv': {
                'channels': DiscreteChoice(sgconv_channels),
                'kernel_size': DiscreteChoice(sgconv_kernel_sizes)
            },
            
            'sep_conv1d': {
                'kernel_size': DiscreteChoice(conv1d_kernel_sizes)
            }
        }, repeat_times=list(range(min_layers, max_layers + 1)), share_arch=False)
    })

    return ConfigSearchSpace(
        GPT2LMHeadModel, 
        single_op_ss, 
        hf_config=hf_config, 
        model_creation_attempts=20,
        seed=seed
    )

