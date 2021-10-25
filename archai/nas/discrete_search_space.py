# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABCMeta, abstractmethod
import torch.nn as nn


class DiscreteSearchSpace(metaclass=ABCMeta):
    @abstractmethod
    def random_sample(self):
        '''Uniform random sample an architecture (nn.Module)'''
        pass

    @abstractmethod
    def get_neighbors(self, arch:nn.Module):
        '''Return the neighbors (some definition of neighborhood) of an architecture'''
        pass


