# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock

import numpy as np
from overrides import overrides

from archai.discrete_search.api.predictor import MeanVar, Predictor


class MyPredictor(Predictor):
    def __init__(self) -> None:
        super().__init__()

    @overrides
    def fit(self, encoded_archs: np.ndarray, y: np.ndarray) -> None:
        return MagicMock()

    @overrides
    def predict(self, encoded_archs: np.ndarray) -> MeanVar:
        return MeanVar(mean=0.0, var=0.0)


def test_predictor():
    predictor = MyPredictor()

    encoded_archs = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2])

    # Assert that mocked methods run and returns proper values
    assert predictor.fit(encoded_archs, y)

    preds = predictor.predict(encoded_archs)
    assert isinstance(preds, MeanVar)
    assert preds.mean == 0.0
    assert preds.var == 0.0
