# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import List

import numpy as np
from overrides import EnforceOverrides

from archai.api.archai_model import ArchaiModel
from archai.api.search_space import SearchSpace


class DiscreteSearchSpace(SearchSpace, EnforceOverrides):
    """Abstract base class for Discrete Search Spaces. Search spaces
    represent all considered architectures of a given task.

    Subclasses of this base class should implement `random_sample`.

    """

    @abstractmethod
    def random_sample(self) -> ArchaiModel:
        """Randomly samples an architecture from the search spaces.

        Returns:
            ArchaiModel: Sampled architecture

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
            np.ndarray: Fixed-length vector representation of `arch`.

        """
