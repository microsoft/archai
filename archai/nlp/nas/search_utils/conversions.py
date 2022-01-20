# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Conversions that enables changing from data types throughout the search.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Matching indexes from the optimization parameters tuple
LOWER_BOUND_IDX = 0
UPPER_BOUND_IDX = 1
PER_LAYER_IDX = 2


def params_to_bound(opt_params: Dict[str, Tuple[int, int, bool]],
                    is_lower_bound: Optional[bool] = True) -> List[Any]:
    """Converts a dictionary of optimization parameters to a bound list.

    Args:
        opt_params: Parameters that are currently being optimized.
        is_lower_bound: Whether should convert to lower or upper bounds.

    Returns:
        (List[Any]): List of bounds.

    """

    # Gathers the appropriate index depending on selected bound
    index = LOWER_BOUND_IDX if is_lower_bound else UPPER_BOUND_IDX

    # Retrieves the maximum number of layers
    max_n_layer = opt_params['n_layer'][UPPER_BOUND_IDX]

    # Iterates through all optimization parameters
    # and calculates the corresponding bound
    bound = []
    for p in opt_params.values():
        # Checks whether current parameter spans for every layer
        if not p[PER_LAYER_IDX]:
            bound += [p[index]]
        else:
            bound += [p[index]] * max_n_layer

    return bound


def position_to_config(positions: np.array,
                       opt_params: Dict[str, Tuple[int, int, bool]]) -> Dict[str, Any]:
    """Converts an array of positions to a configuration dictionary.

    Args:
        positions: Array of positions retrieved from the search space.
        opt_params: Parameters that are currently being optimized.

    Returns:
        (Dict[str, Any]): Configuration dictionary.

    """

    # Removes the extra dimension from array of positions
    positions = np.squeeze(positions, -1)

    # Retrieves the maximum number of layers
    max_n_layer = opt_params['n_layer'][UPPER_BOUND_IDX]

    # Defines an empty index for iteration
    # and an empty configuration object
    param_idx = 0
    config = {}

    # Iterates through all parameters that are being optimized
    for k, v in opt_params.items():
        # Checks whether current parameter spans for every layer
        if not v[PER_LAYER_IDX]:
            config[k] = np.round(positions[param_idx]).astype(int)
            param_idx += 1
        else:
            config[k] = np.round(positions[param_idx: param_idx + max_n_layer]).astype(int).tolist()
            param_idx += max_n_layer

    # Performs a last iteration to remove additional
    # values that are above current number of layers
    for k, v in config.items():
        if isinstance(v, list):
            config[k] = config[k][:config['n_layer']]

    return config
