# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import NamedTuple
from overrides.enforce import EnforceOverrides

import numpy as np


class MeanVar(NamedTuple):
    """Predictive mean and variance estimates from
    a surrogate model (`Predictor`)."""    
    mean: np.ndarray
    var: np.ndarray


class Predictor(EnforceOverrides):
    
    @abstractmethod
    def fit(self, encoded_archs: np.ndarray, y: np.ndarray) -> None:
        """Fits a the predictor model using a an array of encoded architecture (N, #features)
        and a possibly multidimensional array of targets y (N, #targets).

        Args:
            encoded_archs (np.ndarray): (N, #features) numpy array
            y (np.ndarray): (N, #targets) numpy array
        """        

    @abstractmethod
    def predict(self, encoded_archs: np.ndarray) -> MeanVar:
        """Predicts the performance of an array of architec encoded by 
        by a subclass implementation of `BayesOptSearchSpaceBase.encode`

        Args:
            encoded_archs (np.ndarray): Array of encoded architectyres

        Returns:
            MeanVar: Named tuple with `mean` (N, #targets) and `var` (N, #targets) arrays.
        """
