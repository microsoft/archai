from typing import List, Tuple, Union
from itertools import product

import numpy as np
import torch


def split_heads(tensor, num_heads, attn_head_size):
    """
    Splits hidden_size dim into attn_head_size and num_heads
    """
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)


def merge_heads(tensor, num_heads, attn_head_size):
    """
    Merges attn_head_size dim and num_attn_heads dim into hidden_size
    """
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
    return tensor.view(new_shape)


def make_asso_map(input_ids, mask):
    assert mask is not None

    hc_attn = (input_ids.unsqueeze(-1) == input_ids.unsqueeze(1)).float()
    diag_idx = torch.eye(*input_ids.shape[1:]).bool()
    
    hc_attn[:, diag_idx] = 0
    hc_attn *= mask.unsqueeze(-1) * mask.unsqueeze(1)
    hc_attn /= (hc_attn.sum(-1, keepdim=True) + 1e-6)
    
    return hc_attn


def make_broadcast_map(input_ids, mask, eos_id=103):
    T = input_ids.shape[1]
    eos_map = (input_ids == eos_id).float()
    eos_map = eos_map.unsqueeze(1).expand(-1, T, -1)
    eos_mapp = eos_map * (mask.unsqueeze(-1) * mask.unsqueeze(1))
    eos_map = eos_mapp / (eos_map.sum(dim=-1, keepdim=True) + 1e-6)
    
    return eos_map


def get_attn_head_simplex(total_attn_heads: Union[int, List[int]],
                          ops_list: List[str],
                          grid_scale: int = 3) -> List[Tuple]:
    if not isinstance(total_attn_heads, (list, tuple)):
        total_attn_heads = [total_attn_heads]
    
    n_ops = len(ops_list)
    grid = [t for t in product(*[range(grid_scale) for _ in range(n_ops)])]
    grid = grid[1:] # Removes point (0, 0, ..., 0)

    simplex = np.unique(
        np.array(grid) / np.sum(grid, axis=1, keepdims=True), axis=0
    )

    # Stores valid allocations (sum(heads) == total_heads)
    filtered_simplex = []
    
    for total_heads in total_attn_heads:
        heads = np.round(total_heads * simplex)
        filtered_simplex.append(simplex[heads.sum(axis=1) == total_heads])
    
    filtered_simplex = np.concatenate(filtered_simplex, axis=0)
    filtered_simplex = [tuple(a) for a in np.unique(filtered_simplex, axis=0)]
    
    return [
        tuple([(op_name, float(item)) for op_name, item in zip(ops_list, alloc)])
        for alloc in filtered_simplex
    ]
