# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABCMeta, abstractmethod
from typing import List
from collections import namedtuple
from overrides.enforce import EnforceOverrides

import numpy as np

MeanVar = namedtuple("MeanVar", "mean var")


class PredictiveFunction(EnforceOverrides, metaclass=ABCMeta):
    @abstractmethod
    def fit(self, encoded_archs: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, encoded_archs: np.ndarray) -> List[MeanVar]:
        """Predicts the performance of an array of architec encoded by 
        by a subclass implementation of `BayesOptSearchSpaceBase.encode`

        Args:
            encoded_archs (np.ndarray): Array of encoded architectyres

        Returns:
            List[MeanVar]: List of tuples with mean and variance for each prediction
        """        
        ''' Array of architectures encoded by a subclass of `BayesOptSearchSpaceBase`'''
        pass

