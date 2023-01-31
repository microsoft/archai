# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import NamedTuple

import numpy as np
from overrides import EnforceOverrides


class MeanVar(NamedTuple):
    """Predictive mean and variance estimates from a surrogate model (`Predictor`)."""

    mean: np.ndarray
    var: np.ndarray


class Predictor(EnforceOverrides):
    """Abstract class for a predictor model.

    This class can be used to predict the performance of a model given its architecture.
    The class enforces implementation of two methods: `fit` and `predict`.

    Note:
        This class is inherited from `EnforceOverrides` and any overridden methods in the
        subclass should be decorated with `@overrides` to ensure they are properly overridden.

    """

    @abstractmethod
    def fit(self, encoded_archs: np.ndarray, y: np.ndarray) -> None:
        """Fit a predictor model using an array of encoded architectures (N, #features)
        and a multi-dimensional array of targets (N, #targets).

        Args:
            encoded_archs: (N, #features) numpy array.
            y: (N, #targets) numpy array.

        """

        pass

    @abstractmethod
    def predict(self, encoded_archs: np.ndarray) -> MeanVar:
        """Predict the performance of an array of architectures encoded by
        by a subclass implementation of `BayesOptSearchSpaceBase.encode()`.

        Args:
            encoded_archs: Array of encoded architectures.

        Returns:
            Named tuple with `mean` (N, #targets) and `var` (N, #targets) arrays.

        """

        pass
