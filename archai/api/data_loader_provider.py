# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import Any

from overrides import EnforceOverrides


class DataLoaderProvider(EnforceOverrides):
    """Abstract class for data loader providers.

    The `DataLoaderProvider` class provides an abstract interface for creating data loaders
    for a given dataset. The user is required to implement the `get_data_loader` method.
    This method should take in a dataset and return a data loader suitable for that dataset.

    Note:
        This class is inherited from `EnforceOverrides` and any overridden methods in the
        subclass should be decorated with `@overrides` to ensure they are properly overridden.

    Examples:
        >>> class MyDataLoaderProvider(DataLoaderProvider):
        >>>     def __init__(self) -> None:
        >>>         super().__init__()
        >>>
        >>>     @overrides
        >>>     def get_data_loader(self, dataset: Any) -> Any:
        >>>         return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    """

    def __init__(self) -> None:
        """Initialize the data loader provider."""

        pass

    @abstractmethod
    def get_data_loader(self, dataset: Any) -> Any:
        """Get a data loader based on the input dataset.

        This method should take in a dataset and return a data loader suitable for that dataset.

        Args:
            dataset: Input dataset (type of dataset should be specified by the user
                in the implementation of this method).

        Returns:
            Data loader based on the input dataset.

        """

        pass
