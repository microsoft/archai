# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABCMeta, abstractmethod
from typing import List, Union
from overrides.enforce import EnforceOverrides

import torch
import numpy as np
from archai.nas.arch_meta import ArchWithMetaData


class DiscreteSearchSpaceBase(EnforceOverrides, metaclass=ABCMeta):
    ''' Base class for Discrete Search Spaces. '''

    @abstractmethod
    def get(self, idx_vector: List[float]) -> torch.nn.Module:
        ''' Gets an architecture from the search space using `idx_vector`. '''
        pass

    @abstractmethod
    def save_arch(self, model: torch.nn.Module, path: str) -> None:
        pass

    @abstractmethod
    def load_arch(self, path: str) -> torch.nn.Module:
        pass


class EvolutionarySearchSpaceBase(DiscreteSearchSpaceBase, EnforceOverrides):
    ''' Base class for Discrete Search Spaces compatible with Evolutionary search algorithms. '''

    @abstractmethod
    def mutate(self, arch: torch.nn.Module) -> ArchWithMetaData:
        ''' Mutates an architecture from this search space '''
        pass

    @abstractmethod
    def crossover(self, arch_list: List[torch.nn.Module]) -> torch.nn.Module:
        ''' Combines a list of architectures into one new architecture '''
        pass


class BayesOptSearchSpaceBase(DiscreteSearchSpaceBase, EnforceOverrides):
    ''' Base class for Discrete Search Spaces compatible with Bayesian Optimization search algorithms. '''
    
    @abstractmethod
    def encode(self, arch: ArchWithMetaData) -> Union[np.ndarray, torch.Tensor]:
        ''' Encodes an architecture from the search space ''' 


class RLSearchSpaceBase(DiscreteSearchSpaceBase, EnforceOverrides):
    ''' Base class for Discrete Search Spaces compatible with Reinforcement Learning search algorithms. '''

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def reset(self):
        pass