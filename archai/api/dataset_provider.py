# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import Any

from overrides import EnforceOverrides


class DatasetProvider(EnforceOverrides):
    """Abstract class for dataset provider."""

    def __init__(self) -> None:
        """Initializes dataset provider."""

        pass

    @abstractmethod
    def get_train_dataset(self) -> Any:
        """Get a training dataset.

        This function needs to be overriden as any logic can be applied to
        get the dataset.

        Returns:
            An instance of a training dataset, regardless of its structure.

        Examples:
            >>> return torchvision.datasets.MNIST(train=True)

        """

        pass

    @abstractmethod
    def get_val_dataset(self) -> Any:
        """Get a validation dataset.

        This function needs to be overriden as any logic can be applied to
        get the dataset. If a validation dataset is not available, users
        can override and return the training dataset.

        Returns:
            An instance of a validation dataset, regardless of its structure.

        Examples:
            >>> return torchvision.datasets.MNIST(train=False)

        """

        pass

    @abstractmethod
    def get_test_dataset(self) -> Any:
        """Get a testing dataset.

        This function needs to be overriden as any logic can be applied to
        get the dataset. If a testing dataset is not available, users
        can override and return the training/validation dataset.

        Returns:
            An instance of a testing dataset, regardless of its structure.

        Examples:
            >>> return torchvision.datasets.MNIST(train=False)

        """

        pass
