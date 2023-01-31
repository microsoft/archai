# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import pytest
from overrides import overrides

from archai.discrete_search.api.predictor import MeanVar, Predictor


@pytest.fixture
def surrogate_model(search_objectives):
    class DummyPredictor(Predictor):
        def __init__(self, n_objs: int, seed1: int = 10, seed2: int = 20) -> None:
            self.n_objs = n_objs
            self.mean_rng = np.random.RandomState(seed1)
            self.var_rng = np.random.RandomState(seed2)

        @overrides
        def fit(self, encoded_archs: np.ndarray, y: np.ndarray) -> None:
            pass

        @overrides
        def predict(self, encoded_archs: np.ndarray) -> MeanVar:
            n = len(encoded_archs)

            return MeanVar(self.mean_rng.random(size=(n, self.n_objs)), self.var_rng.random(size=(n, self.n_objs)))

    return DummyPredictor(len(search_objectives.exp_objs))
