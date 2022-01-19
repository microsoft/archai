# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

import numpy as np

from opytimizer.spaces import SearchSpace
from opytimizer.core import Function
from opytimizer.optimizers.swarm import PSO

from opytimizer import Opytimizer


#
LOWER_BOUND_IDX = 0
UPPER_BOUND_IDX = 1
PER_LAYER_IDX = 2


class Searcher:
    """
    """

    def __init__(self, params, n_agents=1):
        """
        """

        #
        self._assert_input_params(params)

        #
        if 'n_layer' not in params:
            params['n_layer'] = (1, 1, False)

        #
        self.params = params
        self.max_n_layer = params['n_layer'][1]

        #
        lower_bound = self._calculate_bound(params, is_lower_bound=True)
        upper_bound = self._calculate_bound(params, is_lower_bound=False)

        #
        assert len(lower_bound) == len(upper_bound)
        n_variables = len(lower_bound)

        #
        space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
        function = Function(lambda x: 0)
        heuristic = PSO()

        #
        self.task = Opytimizer(space, heuristic, function)

    def _assert_input_params(self, params):
        """
        """

        for k, v in params.items():
            assert v[LOWER_BOUND_IDX] <= v[UPPER_BOUND_IDX], f'{k}: lower_bound should <= upper_bound'

    def _calculate_bound(self, params, is_lower_bound=True):
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
                bound += [p[index]] * self.max_n_layer

        return bound

    def params_to_config(self, x):
        """
        """

        #
        x = np.squeeze(x, -1)

        #
        param_idx = 0
        config = {}

        #
        for k, v in self.params.items():
            if not v[PER_LAYER_IDX]:
                config[k] = int(x[param_idx])
                param_idx += 1
            else:
                config[k] = x[param_idx: param_idx + self.max_n_layer].astype(int).tolist()
                param_idx += self.max_n_layer

        #
        for k, v in config.items():
            if isinstance(v, list):
                config[k] = config[k][:config['n_layer']]

        return config

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
        'n_layer': (1, 4, False),
        'n_head': (2, 8, True),
        'd_model': (32, 512, False)
    }
    
    h = Searcher(params)
    h.run(n_iterations=100)

    # h.params_to_config(h.task.space.agents[0].position)