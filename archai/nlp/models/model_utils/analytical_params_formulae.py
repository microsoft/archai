# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Formulae to calculate the number of Transformer-based parameters.
"""

from typing import Any, Dict, List, Tuple

from archai.common import utils


def _get_hyperparams(model_config: Dict[str, Any]) -> Tuple[int, int, int, List[int], List[int], List[int]]:
    n_layer = model_config['n_layer']
    n_token = model_config.get('n_token', None) or model_config.get('vocab_size', None)
    tgt_len = model_config.get('tgt_len', None) or model_config.get('n_positions', None)

    d_model = model_config['d_model']
    d_embed = model_config.get('d_embed', None) or d_model
    d_inner = utils.map_to_list(model_config['d_inner'], n_layer)

    n_head = utils.map_to_list(model_config['n_head'], n_layer)
    d_head = [d_model // n_h for n_h in n_head]

    return n_token, tgt_len, d_model, d_embed, d_inner, n_head, d_head


def get_params_hf_codegen_formula(model_config: Dict[str, Any]) -> Dict[str, Any]:
    n_token, _, d_model, _, d_inner, n_head, d_head = _get_hyperparams(model_config)

    params = {
        'embedding': 0,
        'attention': 0,
        'ff': 0,
        'layer_norm': 0,
        'non_embedding': 0,
        'total': 0
    }
    
    # Embedding
    params['embedding'] = n_token * d_model

    for _ in range(len(n_head)):
        # Attention
        # QKV
        params['attention'] += d_model * 3 * n_head[0] * d_head[0]
        # Projection
        params['attention'] += n_head[0] * d_head[0] * d_model

        # Feed-forward
        params['ff'] += (d_model * d_inner[0] + d_inner[0]) + (d_inner[0] * d_model + d_model)

        # Layer normalization
        params['layer_norm'] += 2 * d_model

    # Layer normalization (final)
    params['layer_norm'] += 2 * d_model

    params['non_embedding'] = params['attention'] + params['ff'] + params['layer_norm']
    params['total'] = params['embedding'] + params['non_embedding']

    return params


def get_params_hf_gpt2_formula(model_config: Dict[str, Any]) -> Dict[str, Any]:
    n_token, tgt_len, d_model, _, d_inner, n_head, d_head = _get_hyperparams(model_config)

    params = {
        'embedding': 0,
        'attention': 0,
        'ff': 0,
        'layer_norm': 0,
        'non_embedding': 0,
        'total': 0
    }
    
    # Embedding
    params['embedding'] = n_token * d_model + tgt_len * d_model

    for _ in range(len(n_head)):
        # Attention
        # QKV
        params['attention'] += d_model * 3 * n_head[0] * d_head[0] + 3 * n_head[0] * d_head[0]
        # Projection
        params['attention'] += n_head[0] * d_head[0] * d_model + d_model

        # Feed-forward
        params['ff'] += (d_model * d_inner[0] + d_inner[0]) + (d_inner[0] * d_model + d_model)

        # Layer normalization
        params['layer_norm'] += 2 * d_model * 2

    # Layer normalization (final)
    params['layer_norm'] += 2 * d_model

    params['non_embedding'] = params['attention'] + params['ff'] + params['layer_norm']
    params['total'] = params['embedding'] + params['non_embedding']

    return params


def get_params_hf_gpt2_flex_formula(model_config: Dict[str, Any]) -> Dict[str, Any]:
    n_token, tgt_len, d_model, _, d_inner, n_head, d_head = _get_hyperparams(model_config)

    params = {
        'embedding': 0,
        'attention': 0,
        'ff': 0,
        'layer_norm': 0,
        'non_embedding': 0,
        'total': 0
    }
    
    # Embedding
    params['embedding'] = n_token * d_model + tgt_len * d_model

    for d_i, n_h, d_h in zip(d_inner, n_head, d_head):
        # Attention
        # QKV
        params['attention'] += d_model * 3 * n_h * d_h + 3 * n_h * d_h
        # Projection
        params['attention'] += n_h * d_h * d_model + d_model

        # Feed-forward
        params['ff'] += (d_model * d_i + d_i) + (d_i * d_model + d_model)

        # Layer normalization
        params['layer_norm'] += 2 * d_model * 2

    # Layer normalization (final)
    params['layer_norm'] += 2 * d_model

    params['non_embedding'] = params['attention'] + params['ff'] + params['layer_norm']
    params['total'] = params['embedding'] + params['non_embedding']

    return params


def get_params_hf_opt_formula(model_config: Dict[str, Any]) -> Dict[str, Any]:
    n_token, tgt_len, d_model, _, d_inner, n_head, d_head = _get_hyperparams(model_config)

    params = {
        'embedding': 0,
        'attention': 0,
        'ff': 0,
        'layer_norm': 0,
        'non_embedding': 0,
        'total': 0
    }
    
    # Embedding
    params['embedding'] = n_token * d_model + tgt_len * d_model

    for _ in range(len(n_head)):
        # Attention
        # QKV
        params['attention'] += d_model * 3 * n_head[0] * d_head[0] + 3 * n_head[0] * d_head[0]
        # Projection
        params['attention'] += n_head[0] * d_head[0] * d_model + d_model

        # Feed-forward
        params['ff'] += (d_model * d_inner[0] + d_inner[0]) + (d_inner[0] * d_model + d_model)

        # Layer normalization
        params['layer_norm'] += 2 * d_model * 2

    # Layer normalization (final)
    params['layer_norm'] += 2 * d_model

    params['non_embedding'] = params['attention'] + params['ff'] + params['layer_norm']
    params['total'] = params['embedding'] + params['non_embedding']

    return params


def get_params_hf_transfo_xl_formula(model_config: Dict[str, Any]) -> Dict[str, Any]:
    n_token, _, d_model, d_embed, d_inner, n_head, d_head = _get_hyperparams(model_config)
    div_val = model_config['div_val']
    cutoffs = model_config['cutoffs'] + [n_token]
    cutoff_ends = [0] + cutoffs
    n_clusters = len(cutoffs) - 1

    params = {
        'embedding': 0,
        'softmax': 0,
        'attention': 0,
        'ff': 0,
        'non_embedding': 0,
        'total': 0
    }
    
    # Embedding
    # Handles number of parameters for different div_val
    if div_val == 1:
        params['embedding'] += n_token * d_embed
        if d_embed != d_model:
            params['embedding'] += d_model * d_embed
    else:
        for i in range(len(cutoffs)):
            l_idx, r_idx = cutoff_ends[i], cutoff_ends[i+1]
            d_emb_i = d_embed // (div_val ** i)

            params['embedding'] += (r_idx - l_idx) * d_emb_i
            params['embedding'] += d_model * d_emb_i

    for _ in range(len(n_head)):
        # Attention
        # QKV
        params['attention'] += d_model * 3 * n_head[0] * d_head[0]
        # Positional encoding
        params['attention'] += d_model * n_head[0] * d_head[0]
        # Projection
        params['attention'] += n_head[0] * d_head[0] * d_model
        # r_w_bias and r_r_bias
        params['attention'] += (n_head[0] * d_head[0]) * 2
        # Layer normalization
        params['attention'] += 2 * d_model

        # PositionwiseFF
        params['ff'] += (d_model * d_inner[0] + d_inner[0]) + (d_inner[0] * d_model + d_model) + 2 * d_model

    # Softmax
    # Clusters
    params['softmax'] += n_clusters * d_embed + n_clusters

    # Handles number of parameters for different div_val
    if div_val == 1:
        for i in range(len(cutoffs)):
            if d_model != d_embed:
                params['softmax'] += d_model * d_embed
        params['softmax'] += d_embed * n_token + n_token
    else:
        for i in range(len(cutoffs)):
            l_idx, r_idx = cutoff_ends[i], cutoff_ends[i+1]
            d_emb_i = d_embed // (div_val ** i)

            params['softmax'] += d_model * d_emb_i
            params['softmax'] += d_emb_i * (r_idx - l_idx) + (r_idx - l_idx)

    params['non_embedding'] = params['attention'] + params['ff']
    params['total'] = params['embedding'] + params['non_embedding'] + params['softmax']

    return params


def get_params_mem_transformer_formula(model_config: Dict[str, Any]) -> Dict[str, Any]:
    n_token, _, d_model, d_embed, d_inner, n_head, d_head = _get_hyperparams(model_config)
    div_val = model_config['div_val']
    cutoffs = model_config['cutoffs'] + [n_token]
    cutoff_ends = [0] + cutoffs
    n_clusters = len(cutoffs) - 1
    tie_projs = model_config['tie_projs']
    tie_weight = model_config['tie_weight']

    params = {
        'embedding': 0,
        'softmax': 0,
        'attention': 0,
        'ff': 0,
        'layer_norm': 0,
        'non_embedding': 0,
        'total': 0
    }
    
    # Embedding
    # Handles number of parameters for different div_val
    if div_val == 1:
        params['embedding'] += n_token * d_embed
        if d_embed != d_model:
            params['embedding'] += d_model * d_embed
    else:
        for i in range(len(cutoffs)):
            l_idx, r_idx = cutoff_ends[i], cutoff_ends[i+1]
            d_emb_i = d_embed // (div_val ** i)

            params['embedding'] += (r_idx - l_idx) * d_emb_i
            params['embedding'] += d_model * d_emb_i

    for d_i, n_h, d_h in zip(d_inner, n_head, d_head):
        # Attention
        # QKV
        params['attention'] += d_model * 3 * n_h * d_h
        # Positional encoding
        params['attention'] += d_model * n_h * d_h
        # Projection
        params['attention'] += n_h * d_h * d_model
        # r_w_bias and r_r_bias
        params['attention'] += (n_h * d_h) * 2

        # CoreNet (feed-forward)
        params['ff'] += (d_model * d_i + d_i) + (d_i * d_model + d_model)
        
        # Layer normalization
        params['layer_norm'] += (2 * d_model) * 2

    # Softmax
    # Clusters
    params['softmax'] += n_clusters * d_embed + n_clusters

    # Shared output projections
    params['softmax'] += sum([d_embed * (d_embed // (div_val ** i)) for i in range(len(cutoffs))])

    # Handles number of parameters for different div_val
    if div_val == 1:
        if d_embed != d_model:
            params['softmax'] += sum([d_model * d_embed if not tie else 0 for tie in tie_projs])
        params['softmax'] += n_token
    else:
        for i in range(len(cutoffs)):
            l_idx, r_idx = cutoff_ends[i], cutoff_ends[i+1]
            d_emb_i = d_embed // (div_val ** i)

            params['softmax'] += d_model * d_emb_i if not tie_projs[i] else 0
            params['softmax'] += (r_idx - l_idx)
            
            if not tie_weight:
                params['softmax'] += (r_idx - l_idx) * d_emb_i

    params['non_embedding'] = params['attention'] + params['ff'] + params['layer_norm']
    params['total'] = params['embedding'] + params['non_embedding'] + params['softmax']

    return params
