# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import List, Union
from overrides.enforce import EnforceOverrides

import torch
import numpy as np
from archai.discrete_search.api.model import NasModel


class DiscreteSearchSpace(EnforceOverrides):
    """Abstract base class for Discrete Search Spaces. Search spaces
    represent all considered architectures of a given task.

    Subclasses of this base class should implement `random_sample`,
    `save_arch`, `load_arch`, `save_model_weights` and `load_model_weights`.
    """    

    @abstractmethod
    def random_sample(self) -> NasModel:
        """Randomly samples an architecture from the search spaces.

        Returns:
            NasModel: Sampled architecture
        """        

    @abstractmethod
    def save_arch(self, model: NasModel, path: str) -> None:
        """Saves an the architecture of `model` into a file without saving
        model weights.

        Args:
            model (NasModel): Sampled model from the search space
            path (str): Filepath
        """

    @abstractmethod
    def load_arch(self, path: str) -> NasModel:
        """Loads an architecture saved using `DiscreteSearchSpace.save_arch` from a file.

        Args:
            path (str): Architecture file path

        Returns:
            NasModel: Loaded model
        """        

    @abstractmethod
    def save_model_weights(self, model: NasModel, path: str) -> None:
        """Saves weights of a model from the search space.

        Args:
            model (NasModel): Model
            path (str): Model weights file path
        """        

    @abstractmethod
    def load_model_weights(self, model: NasModel, path: str) -> None:
        """Loads a weight file (create using `DiscreteSearchSpace.save_model_weights`)
        into a model of the same architecture.
        
        Args:
            model (NasModel): Target model
            path (str): Model weights file path.
        """


class EvolutionarySearchSpace(DiscreteSearchSpace, EnforceOverrides):
    """Abstract base class for evolutionary algo-compatible discrete search spaces.
       Subclasses are expected to implement `mutate` and `crossover` methods.
    """    

    @abstractmethod
    def mutate(self, arch: NasModel) -> NasModel:
        """Mutates an architecture from the search space. This method should not alter
          the base model architecture directly, only generate a new one.

        Args:
            arch (NasModel): Base model

        Returns:
            NasModel: Mutated model
        """        

    @abstractmethod
    def crossover(self, arch_list: List[NasModel]) -> NasModel:
        """Combines a list of architectures into a new one.

        Args:
            arch_list (List[NasModel]): List of architectures

        Returns:
            NasModel: Resulting model
        """        


class BayesOptSearchSpace(DiscreteSearchSpace, EnforceOverrides):
    """Abstract base class for discrete search spaces compatible with Bayesian Optimization algorithms.
       Subclasses are expected to implement `encode`.
    """    
    
    @abstractmethod
    def encode(self, arch: NasModel) -> np.ndarray:
        """Encodes an architecture into a fixed-length vector representation.

        Args:
            arch (NasModel): Model from the search space

        Returns:
            np.ndarray: Fixed-length vector representation of `arch`
        """


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
