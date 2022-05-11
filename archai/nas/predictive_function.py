# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABCMeta, abstractmethod
from typing import List
from collections import namedtuple
from overrides.enforce import EnforceOverrides

import torch.nn as nn
from archai.nas.arch_meta import ArchWithMetaData

MeanVar = namedtuple("MeanVar", "mean var")


class PredictiveFunction(EnforceOverrides, metaclass=ABCMeta):

    @abstractmethod
    def predict(self, arch:ArchWithMetaData)->MeanVar:
        ''' Predictive function to be used with BO methods
        like Bananas '''
        pass

