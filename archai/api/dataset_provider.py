# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import Any

from overrides import EnforceOverrides


class DatasetProvider(EnforceOverrides):
    """Abstract class for dataset providers.

    This class serves as a base for implementing dataset providers that can return
    training, validation and testing datasets. The class enforces implementation
    of three methods: `get_train_dataset`, `get_val_dataset` and `get_test_dataset`.
    These methods should return an instance of the  respective dataset, regardless of
    its structure.

    Note:
        This class is inherited from `EnforceOverrides` and any overridden methods in the
        subclass should be decorated with `@overrides` to ensure they are properly overridden.

    Examples:
        >>> class MyDatasetProvider(DatasetProvider):
        >>>     def __init__(self) -> None:
        >>>         super().__init__()
        >>>
        >>>     @overrides
        >>>     def get_train_dataset(self) -> Any:
        >>>         return torchvision.datasets.MNIST(train=True)
        >>>
        >>>     @overrides
        >>>     def get_val_dataset(self) -> Any:
        >>>         return torchvision.datasets.MNIST(train=False)
        >>>
        >>>     @overrides
        >>>     def get_test_dataset(self) -> Any:
        >>>         return torchvision.datasets.MNIST(train=False)


    """

    def __init__(self) -> None:
        """Initialize the dataset provider."""

        pass

    @abstractmethod
    def get_train_dataset(self) -> Any:
        """Get a training dataset.

        Returns:
            An instance of a training dataset.

        """

        pass

    @abstractmethod
    def get_val_dataset(self) -> Any:
        """Get a validation dataset.

        Returns:
            An instance of a validation dataset, or the training dataset if
            validation dataset is not available.

        """

        pass

    @abstractmethod
    def get_test_dataset(self) -> Any:
        """Get a testing dataset.

        Returns:
            An instance of a testing dataset, or the training/validation
            dataset if testing dataset is not available.

        """

        pass
