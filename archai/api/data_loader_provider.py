# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import Any

from overrides import EnforceOverrides


class DataLoaderProvider(EnforceOverrides):
    """Abstract class for data loader provider."""

    def __init__(self) -> None:
        """Initializes data loader provider."""

        pass

    @abstractmethod
    def get_data_loader(self, dataset: Any) -> Any:
        """Get a data loader based on the input dataset.

        This function needs to be overriden as any logic can be applied to
        get the data loader.

        Args:
            dataset: Input dataset.

        Returns:
            Data loader based on the input dataset.

        Examples:
            >>> return torch.utils.data.DataLoader(dataset)

        """

        pass
