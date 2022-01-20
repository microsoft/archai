# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Provides a wrapper for conducting neural architecture searches.
"""

from typing import Dict, Optional, Tuple

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.spaces import SearchSpace

from archai.nlp.nas.search_utils.conversions import (LOWER_BOUND_IDX,
                                                     UPPER_BOUND_IDX,
                                                     params_to_bound,
                                                     position_to_config)
from archai.nlp.nas.search_utils.heuristics.heuristic_loader import load_heuristic_from_args
from archai.nlp.nas.search_utils.objectives import zero_cost_objective


class ArchaiSearcher:
    """Creates a Trainer-like utility for searches and allows to be customized.

    """

    def __init__(self,
                 model_type: str,
                 heuristic_type: str,
                 opt_params: Dict[str, Tuple[int, int, bool]],
                 n_agents: Optional[int] = 1,
                 save_agents: Optional[bool] = True) -> None:
        """Overrides with custom initialization.

        Args:
            model_type: Type of model.
            heuristic_type: Type of heuristic.
            opt_params: Optimization parameters.
            n_agents: Number of agents.
            save_agents: Whether to save all agents into search history.

        """

        # Asserts whether input parameters are valid or not
        self._assert_input_params(opt_params)

        # Always make sure that the number of layers is supplied to
        # the number of optimization parameters
        if 'n_layer' not in opt_params:
            opt_params['n_layer'] = (1, 1, False)

        # Calculates the lower and upper bounds
        lower_bound = params_to_bound(opt_params, is_lower_bound=True)
        upper_bound = params_to_bound(opt_params, is_lower_bound=False)

        # Asserts that the length of bounds are equal
        # and that they should correspond to the number of variables to be optimized
        assert len(lower_bound) == len(upper_bound)
        n_variables = len(lower_bound)

        # Initializes the parameters space and its objective
        space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
        objective = zero_cost_objective(model_type, opt_params)

        # Creates function, heuristic and optimization task
        function = Function(objective)
        heuristic = load_heuristic_from_args(heuristic_type)
        task = Opytimizer(space, heuristic, function, save_agents=save_agents)

        # Attaches important properties
        self.opt_params = opt_params
        self.task = task

    def _assert_input_params(self, params: Dict[str, Tuple[int, int, bool]]) -> None:
        """Asserts whether input optimization parameters are valid or not.

        Args:
            params: Optimization parameters.

        """

        for k, v in params.items():
            assert v[LOWER_BOUND_IDX] <= v[UPPER_BOUND_IDX], f'{k}: lower_bound should <= upper_bound'

    def run(self, n_iterations: Optional[int] = 1) -> None:
        """Runs a search-based task.

        Args:
            n_iterations: Number of optimization iterations.

        """

        # Initializes the search procedure
        self.task.start(n_iterations=n_iterations)


if __name__ == '__main__':
    params = {
        'n_head': (2, 8, True),
        'd_inner': (32, 512, False),
        'n_layer': (3, 12, False)
    }
    
    h = ArchaiSearcher('mem_transformer', 'pso', params, n_agents=10)
    h.run(n_iterations=5)

    print(position_to_config(h.task.space.best_agent.position, h.params))
