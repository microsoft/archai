# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import List

import numpy as np
from overrides import EnforceOverrides

from archai.api.archai_model import ArchaiModel
from archai.api.search_space import SearchSpace


class DiscreteSearchSpace(SearchSpace, EnforceOverrides):
    """Abstract class for discrete search spaces.

    The class enforces implementation of a single method: `random_sample`.

    Note:
        This class is inherited from `EnforceOverrides` and any overridden methods in the
        subclass should be decorated with `@overrides` to ensure they are properly overridden.

    """

    @abstractmethod
    def random_sample(self) -> ArchaiModel:
        """Randomly sample an architecture from the search spaces.

        Returns:
            Sampled architecture.

        """

        pass


class EvolutionarySearchSpace(DiscreteSearchSpace, EnforceOverrides):
    """Abstract class for discrete search spaces compatible with evolutionary algorithms.

    The class enforces implementation of two methods: `mutate` and `crossover`.

    Note:
        This class is inherited from `EnforceOverrides` and any overridden methods in the
        subclass should be decorated with `@overrides` to ensure they are properly overridden.

    """

    @abstractmethod
    def mutate(self, arch: ArchaiModel) -> ArchaiModel:
        """Mutate an architecture from the search space.

        This method should not alter the base model architecture directly,
        only generate a new one.

        Args:
            arch: Base model.

        Returns:
            Mutated model.

        """

        pass

    @abstractmethod
    def crossover(self, arch_list: List[ArchaiModel]) -> ArchaiModel:
        """Combine a list of architectures into a new one.

        Args:
            arch_list: List of architectures.

        Returns:
            Resulting model.

        """

        pass


class BayesOptSearchSpace(DiscreteSearchSpace, EnforceOverrides):
    """Abstract class for discrete search spaces compatible with Bayesian Optimization algorithms.

    The class enforces implementation of a single method: `encode`.

    Note:
        This class is inherited from `EnforceOverrides` and any overridden methods in the
        subclass should be decorated with `@overrides` to ensure they are properly overridden.

    """

    @abstractmethod
    def encode(self, arch: ArchaiModel) -> np.ndarray:
        """Encode an architecture into a fixed-length vector representation.

        Args:
            arch: Model from the search space.

        Returns:
            Fixed-length vector representation of `arch`.

        """

        pass
