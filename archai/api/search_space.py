# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod

from overrides import EnforceOverrides

from archai.api.archai_model import ArchaiModel


class SearchSpace(EnforceOverrides):
    """Abstract class for search spaces.

    This class serves as a base for implementing search spaces. The class enforces
    implementation of four methods: `save_arch`, `load_arch`, `save_model_weights` and
    `load_model_weights`.

    Note:
        This class is inherited from `EnforceOverrides` and any overridden methods in the
        subclass should be decorated with `@overrides` to ensure they are properly overridden.

    Examples:
        >>> class MySearchSpace(SearchSpace):
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


    """

    def __init__(self) -> None:
        """Initialize the search space."""

        pass

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
