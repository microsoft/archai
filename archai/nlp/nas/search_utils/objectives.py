# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Objectives (fitness functions) used to conduct optimization tasks.
"""

from typing import Dict, Tuple

import numpy as np

from archai.nlp.models.model_loader import load_model_from_args
from archai.nlp.nas.search_utils.constraints import measure_parameters
from archai.nlp.nas.search_utils.conversions import position_to_config


def zero_cost_objective(model_type: str,
                        opt_params: Dict[str, Tuple[int, int, bool]]) -> callable:
    """Performs a zero-cost search by calculating the number of decoder parameters,
        latency and peak memory.

    Args:
        model_type: Type of model.
        opt_params: Optimization parameters.

    Returns:
        (callable): The evaluation function itself.

    """

    def f(x: np.array) -> float:
        """Wrapper around the function itself.

        Args:
            x: An array holding the variables that are being searched.

        Returns:
            (float): A floating point holding the fitness function value.

        """

        # Retrieves the default configuration based on model's type
        model_config = load_model_from_args(model_type, cls_type='config').default

        # Converts current position into a configuration dictionary
        # and updates the model's configuration
        config = position_to_config(x, opt_params)
        model_config.update(config)

        # Loads the model
        model = load_model_from_args(model_type, **model_config)

        # Measures its number of parameters
        n_params = measure_parameters(model)

        return n_params

    return f
