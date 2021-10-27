# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABCMeta, abstractmethod
from typing import List
from overrides.enforce import EnforceOverrides

import torch.nn as nn
from archai.nas.arch_meta import ArchWithMetaData


class DiscreteSearchSpace(EnforceOverrides, metaclass=ABCMeta):
    @abstractmethod
    def random_sample(self)->ArchWithMetaData:
        '''Uniform random sample an architecture (nn.Module)'''
        pass

    @abstractmethod
    def get_neighbors(self, arch:ArchWithMetaData)->List[ArchWithMetaData]:
        '''Return the neighbors (some definition of neighborhood) of an architecture'''
        pass

