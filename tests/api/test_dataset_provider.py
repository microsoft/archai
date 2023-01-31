# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any
from unittest.mock import MagicMock

from overrides import overrides

from archai.api.dataset_provider import DatasetProvider


class MyDatasetProvider(DatasetProvider):
    def __init__(self) -> None:
        super().__init__()

    @overrides
    def get_train_dataset(self) -> Any:
        return MagicMock()

    @overrides
    def get_val_dataset(self) -> Any:
        return MagicMock()

    @overrides
    def get_test_dataset(self) -> Any:
        return MagicMock()


def test_my_dataset_provider():
    dataset_provider = MyDatasetProvider()

    train_dataset = dataset_provider.get_train_dataset()
    val_dataset = dataset_provider.get_val_dataset()
    test_dataset = dataset_provider.get_test_dataset()

    assert train_dataset is not None
    assert val_dataset is not None
    assert test_dataset is not None
