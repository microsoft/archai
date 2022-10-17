# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import List, Union
from overrides.enforce import EnforceOverrides

import torch
import numpy as np
from archai.discrete_search.api.model import NasModel


class DiscreteSearchSpace(EnforceOverrides):
    ''' Base class for Discrete Search Spaces. '''

    @abstractmethod
    def random_sample(self) -> NasModel:
        ''' Samples an architecture from the search space''' 

    @abstractmethod
    def save_arch(self, model: NasModel, path: str) -> None:
        ''' Saves the architecture (without model weights) in a file''' 

    @abstractmethod
    def load_arch(self, path: str) -> NasModel:
        ''' Loads an architecture saved using `save_arch`''' 

    @abstractmethod
    def save_model_weights(self, model: NasModel, path: str) -> None:
        ''' Saves model weights''' 

    @abstractmethod
    def load_model_weights(self, model: NasModel, path: str) -> None:
        ''' Loads model weights''' 


class EvolutionarySearchSpace(DiscreteSearchSpace, EnforceOverrides):
    ''' Base class for Discrete Search Spaces compatible with Evolutionary search algorithms. '''

    @abstractmethod
    def mutate(self, arch: NasModel) -> NasModel:
        ''' Mutates an architecture from this search space '''

    @abstractmethod
    def crossover(self, arch_list: List[NasModel]) -> NasModel:
        ''' Combines a list of architectures into a new architecture'''


class BayesOptSearchSpace(DiscreteSearchSpace, EnforceOverrides):
    ''' Base class for Discrete Search Spaces compatible with Bayesian Optimization search algorithms. '''
    
    @abstractmethod
    def encode(self, arch: NasModel) -> np.ndarray:
        ''' Encodes an architecture from the search space ''' 


class RLSearchSpace(DiscreteSearchSpace, EnforceOverrides):
    ''' Base class for Discrete Search Spaces compatible with Reinforcement Learning search algorithms. '''
    
    @abstractmethod
    def get(self, idx_vector: float) -> NasModel:
        ''' Gets a NasModel from the search space using `idx_vector`. '''

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def reset(self):
        pass
