# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import List, Union
from overrides.enforce import EnforceOverrides

import torch
import numpy as np
from archai.discrete_search.api.archai_model import ArchaiModel


class DiscreteSearchSpace(EnforceOverrides):
    """Abstract base class for Discrete Search Spaces. Search spaces
    represent all considered architectures of a given task.

    Subclasses of this base class should implement `random_sample`,
    `save_arch`, `load_arch`, `save_model_weights` and `load_model_weights`.
    """    

    @abstractmethod
    def random_sample(self) -> ArchaiModel:
        """Randomly samples an architecture from the search spaces.

        Returns:
            ArchaiModel: Sampled architecture
        """        

    @abstractmethod
    def save_arch(self, model: ArchaiModel, path: str) -> None:
        """Saves an the architecture of `model` into a file without saving
        model weights.

        Args:
            model (ArchaiModel): Sampled model from the search space
            path (str): Filepath
        """

    @abstractmethod
    def load_arch(self, path: str) -> ArchaiModel:
        """Loads an architecture saved using `DiscreteSearchSpace.save_arch` from a file.

        Args:
            path (str): Architecture file path

        Returns:
            ArchaiModel: Loaded model
        """        

    @abstractmethod
    def save_model_weights(self, model: ArchaiModel, path: str) -> None:
        """Saves weights of a model from the search space.

        Args:
            model (ArchaiModel): Model
            path (str): Model weights file path
        """        

    @abstractmethod
    def load_model_weights(self, model: ArchaiModel, path: str) -> None:
        """Loads a weight file (create using `DiscreteSearchSpace.save_model_weights`)
        into a model of the same architecture.
        
        Args:
            model (ArchaiModel): Target model
            path (str): Model weights file path.
        """


class EvolutionarySearchSpace(DiscreteSearchSpace, EnforceOverrides):
    """Abstract base class for evolutionary algo-compatible discrete search spaces.
       Subclasses are expected to implement `mutate` and `crossover` methods.
    """    

    @abstractmethod
    def mutate(self, arch: ArchaiModel) -> ArchaiModel:
        """Mutates an architecture from the search space. This method should not alter
          the base model architecture directly, only generate a new one.

        Args:
            arch (ArchaiModel): Base model

        Returns:
            ArchaiModel: Mutated model
        """        

    @abstractmethod
    def crossover(self, arch_list: List[ArchaiModel]) -> ArchaiModel:
        """Combines a list of architectures into a new one.

        Args:
            arch_list (List[ArchaiModel]): List of architectures

        Returns:
            ArchaiModel: Resulting model
        """        


class BayesOptSearchSpace(DiscreteSearchSpace, EnforceOverrides):
    """Abstract base class for discrete search spaces compatible with Bayesian Optimization algorithms.
       Subclasses are expected to implement `encode`.
    """    
    
    @abstractmethod
    def encode(self, arch: ArchaiModel) -> np.ndarray:
        """Encodes an architecture into a fixed-length vector representation.

        Args:
            arch (ArchaiModel): Model from the search space

        Returns:
            np.ndarray: Fixed-length vector representation of `arch`
        """


class RLSearchSpace(DiscreteSearchSpace, EnforceOverrides):
    ''' Base class for Discrete Search Spaces compatible with Reinforcement Learning search algorithms. '''
    
    @abstractmethod
    def get(self, idx_vector: float) -> ArchaiModel:
        ''' Gets a ArchaiModel from the search space using `idx_vector`. '''

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def reset(self):
        pass
