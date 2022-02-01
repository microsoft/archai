# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Mathematical formulae used to calculate Transformer-based measurements.
"""

from typing import Any, Dict, List, Tuple


def _get_hyperparams(model_config: Dict[str, Any]) -> Tuple[int, int, int, List[int], List[int], List[int]]:
    n_layer = model_config['n_layer']
    n_token = model_config['n_token']
    tgt_len = model_config['tgt_len']

    d_model = model_config['d_model']
    d_embed = d_model if model_config['d_embed'] < 0 else model_config['d_embed']
    d_inner = model_config['d_inner'][:n_layer]

    n_head = model_config['n_head'][:n_layer]
    d_head = [d_model // n_h for n_h in n_head]

    return n_token, tgt_len, d_model, d_embed, d_inner, n_head, d_head


def get_params_gpt2_formula(model_config: Dict[str, Any]) -> Dict[str, Any]:
    # Gathers hyperparameters
    n_token, tgt_len, d_model, d_embed, d_inner, n_head, d_head = _get_hyperparams(model_config)

    # Defines an empty dictionary of parameters
    params = {
        'embedding': 0,
        'attention': 0,
        'ff': 0,
        'non_embedding': 0,
        'total': 0
    }
    
    # Embedding layer: (n_token * d_embed) + (tgt_len * d_embed)
    params['embedding'] = n_token * d_embed + tgt_len * d_embed

    for d_i, n_h, d_h in zip(d_inner, n_head, d_head):
        # Attention layer
        # QKV: (d_model * 3 * n_head * d_head + 3 * n_head * d_head)
        params['attention'] += d_model * 3 * n_h * d_h + 3 * n_h * d_h

        # Projection: (n_head * d_head * d_model + d_model)
        params['attention'] += n_h * d_h * d_model + d_model

        # Feed-forward layer: (d_model * d_inner + d_inner) + (d_inner * d_model + d_model)
        params['ff'] += (d_model * d_i + d_i) + (d_i * d_model + d_model)

    params['non_embedding'] = params['attention'] + params['ff']
    params['total'] = params['embedding'] + params['non_embedding']

    return params


def get_params_transformer_xl_formula(model_config: Dict[str, Any]) -> Dict[str, Any]:
    # Gathers hyperparameters
    n_token, tgt_len, d_model, d_embed, d_inner, n_head, d_head = _get_hyperparams(model_config)

    # Defines an empty dictionary of parameters
    params = {
        'embedding': 0,
        'softmax': 0,
        'attention': 0,
        'ff': 0,
        'non_embedding': 0,
        'total': 0
    }
    
    # Embedding layer: (n_token * d_embed) + (tgt_len * d_embed)
    params['embedding'] = n_token * d_embed + tgt_len * d_embed

    for d_i, n_h, d_h in zip(d_inner, n_head, d_head):
        # Attention layer
        # QKV: (d_model * 3 * n_head * d_head + 3 * n_head * d_head)
        params['attention'] += d_model * 3 * n_h * d_h

        # Positional encoding: (d_model * n_head * d_head)
        params['attention'] += d_model * n_h * d_h

        # Projection: (n_head * d_head * d_model)
        params['attention'] += n_h * d_h * d_model

        # r_w_bias and r_r_bias: (n_head * d_head) * 2
        params['attention'] += (n_h * d_h) * 2

        # Feed-forward layer
        # Linear: (d_model * d_inner + d_inner) + (d_inner * d_model + d_model)
        params['ff'] += (d_model * d_i + d_i) + (d_i * d_model + d_model)

        # Layer normalization: (d_model * 2)
        params['ff'] += d_model * 2

    params['non_embedding'] = params['attention'] + params['ff']
    params['total'] = params['embedding'] + params['non_embedding']

    return params
