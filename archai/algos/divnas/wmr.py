# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np



class Wmr:
    """ Implements the Randomized Weighted Majority algorithm by Littlestone and Warmuth
    We use the version in Fig 1 in The Multiplicative Weight Update with the gain version """
    def __init__(self, num_items:int, eta:float):
        assert num_items > 0 
        assert eta >= 0.0 and eta <= 0.5
        self._num_items = num_items
        self._eta = eta
        self._weights = self._normalize(np.ones(self._num_items))
        self._round_counter = 0

    @property
    def weights(self):
        return self._weights

    def _normalize(self, weights:np.array)->None:
        return weights / np.sum(weights)

    def update(self, rewards:np.array)->None:
        assert len(rewards.shape) == 1
        assert rewards.shape[0] == self._num_items
        assert np.all(rewards) >= -1 and np.all(rewards) <= 1.0

        # # annealed learning rate
        # self._round_counter += 1
        # eta = self._eta / np.sqrt(self._round_counter)
        eta = self._eta

        self._weights = self._weights * (1.0 + eta * rewards)
        self._weights = self._normalize(self._weights)
        
        
    def sample(self)->int:
        return np.random.choice(self._num_items, p=self._normalize(self._weights))






