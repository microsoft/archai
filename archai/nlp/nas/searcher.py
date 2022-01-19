# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

import numpy as np

from opytimizer.spaces import SearchSpace
from opytimizer.core import Function
from opytimizer.optimizers.evolutionary import GA
from opytimizer.optimizers.swarm import PSO

from opytimizer import Opytimizer

from archai.nlp.nas.objectives import zero_cost_objective


#
LOWER_BOUND_IDX = 0
UPPER_BOUND_IDX = 1
PER_LAYER_IDX = 2


class Searcher:
    """
    """

    def __init__(self, model_type, params, n_agents=1):
        """
        """

        #
        self._assert_input_params(params)

        #
        if 'n_layer' not in params:
            params['n_layer'] = (1, 1, False)
        max_n_layer = params['n_layer'][UPPER_BOUND_IDX]

        #
        lower_bound = self._calculate_bound(params, max_n_layer, is_lower_bound=True)
        upper_bound = self._calculate_bound(params, max_n_layer, is_lower_bound=False)

        #
        assert len(lower_bound) == len(upper_bound)
        n_variables = len(lower_bound)

        #
        space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
        objective = zero_cost_objective(model_type, params, max_n_layer)

        #
        function = Function(objective)
        heuristic = PSO()
        task = Opytimizer(space, heuristic, function)

        #
        self.params = params
        self.max_n_layer = max_n_layer
        self.task = task

    def _assert_input_params(self, params):
        """
        """

        for k, v in params.items():
            assert v[LOWER_BOUND_IDX] <= v[UPPER_BOUND_IDX], f'{k}: lower_bound should <= upper_bound'

    def _calculate_bound(self, params, max_n_layer, is_lower_bound=True):
        """
        """

        #
        index = LOWER_BOUND_IDX if is_lower_bound else UPPER_BOUND_IDX

        #
        bound = []
        for p in params.values():
            if not p[PER_LAYER_IDX]:
                bound += [p[index]]
            else:
                bound += [p[index]] * max_n_layer

        return bound

    def run(self, n_iterations=1):
        """
        """

        self.task.start(n_iterations)





if __name__ == '__main__':
    # params = {
    #     'n_head': [2, 8],
    #     'd_model': [32, 512],
    #     'n_layer': [3, 12],
    # }

    params = {
        'n_head': (2, 8, True),
        'd_model': (32, 512, False),
        'n_layer': (3, 12, False)
    }
    
    h = Searcher('mem_transformer', params, n_agents=25)
    h.run(n_iterations=100)

    from archai.nlp.nas.converter import params_to_config

    print(params_to_config(h.params, h.max_n_layer, h.task.space.best_agent.position))

    # h.params_to_config(h.task.space.agents[0].position)