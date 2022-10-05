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
    def get(self, idx_vector: List[float]) -> ArchWithMetaData:
        ''' Gets a NasModel from the search space using `idx_vector`. '''
        pass

    @abstractmethod
    def save_arch(self, model: ArchWithMetaData, path: str) -> None:
        ''' Saves the architecture (without model weights) in a file''' 
        pass

    @abstractmethod
    def save_model_weights(self, model: ArchWithMetaData, path: str) -> None:
        ''' Saves model weights '''
        pass

    @abstractmethod
    def load_arch(self, path: str) -> ArchWithMetaData:
        ''' Loads an architecture saved using `save_arch`''' 
        pass

    @abstractmethod
    def load_model_weights(self, model: ArchWithMetaData, path: str) -> None:
        '''Loads the weights saved using `save_model_weights` in an architecture. ''' 
        pass


class EvolutionarySearchSpaceBase(DiscreteSearchSpaceBase, EnforceOverrides):
    ''' Base class for Discrete Search Spaces compatible with Evolutionary search algorithms. '''

    @abstractmethod
    def mutate(self, arch: ArchWithMetaData) -> ArchWithMetaData:
        ''' Mutates an architecture from this search space '''
        pass

    @abstractmethod
    def crossover(self, arch_list: List[ArchWithMetaData]) -> ArchWithMetaData:
        ''' Combines a list of architectures into a new architecture'''
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