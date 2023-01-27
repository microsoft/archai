# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import List

import numpy as np
from overrides import EnforceOverrides

from archai.discrete_search.api.archai_model import ArchaiModel


class DiscreteSearchSpace(EnforceOverrides):
    """Abstract class for discrete search spaces.

    This class serves as a base for implementing search spaces. The class enforces
    implementation of five methods: `save_arch`, `load_arch`, `save_model_weights`,
    `load_model_weights` and `random_sample`.

    Note:
        This class is inherited from `EnforceOverrides` and any overridden methods in the
        subclass should be decorated with `@overrides` to ensure they are properly overridden.

    Examples:
        >>> class MyDiscreteSearchSpace(DiscreteSearchSpace):
        >>>     def __init__(self) -> None:
        >>>         super().__init__()
        >>>
        >>>     @overrides
        >>>     def save_arch(self, arch, file_path) -> None:
        >>>         torch.save(arch, file_path)
        >>>
        >>>     @overrides
        >>>     def load_arch(self, file_path) -> ArchaiModel:
        >>>         return torch.load(file_path)
        >>>
        >>>     @overrides
        >>>     def save_model_weights(self, model, file_path) -> None:
        >>>         torch.save(model.state_dict(), file_path)
        >>>
        >>>     @overrides
        >>>     def load_model_weights(self, model, file_path) -> None:
        >>>         model.load_state_dict(torch.load(file_path))
        >>>
        >>>     @overrides
        >>>     def random_sample(self, config) -> ArchaiModel:
        >>>         return ArchaiModel(config)

    """

    @abstractmethod
    def save_arch(self, model: ArchaiModel, file_path: str) -> None:
        """Save an architecture to a file without saving the weights.

        Args:
            model: Model's architecture to save.
            file_path: File path to save the architecture.

        """

        pass

    @abstractmethod
    def load_arch(self, file_path: str) -> ArchaiModel:
        """Load from a file an architecture that was saved using `SearchSpace.save_arch()`.

        Args:
            file_path: File path to load the architecture.

        Returns:
            Loaded model.

        """

        pass

    @abstractmethod
    def save_model_weights(self, model: ArchaiModel, file_path: str) -> None:
        """Save the weights of a model.

        Args:
            model: Model to save the weights.
            file_path: File path to save the weights.

        """

        pass

    @abstractmethod
    def load_model_weights(self, model: ArchaiModel, file_path: str) -> None:
        """Load the weights (created with `SearchSpace.save_model_weights()`) into a model
        of the same architecture.

        Args:
            model: Model to load the weights.
            file_path: File path to load the weights.

        """

        pass

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
