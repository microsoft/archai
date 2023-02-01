# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Optional
from unittest.mock import MagicMock

from overrides import overrides

from archai.api.dataset_provider import DatasetProvider
from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.model_evaluator import (
    AsyncModelEvaluator,
    ModelEvaluator,
)


class MyModelEvaluator(ModelEvaluator):
    def __init__(self) -> None:
        super().__init__()

    @overrides
    def evaluate(self, arch: ArchaiModel, dataset: DatasetProvider, budget: Optional[float] = None) -> float:
        return 0.0


class MyAsyncModelEvaluator(AsyncModelEvaluator):
    def __init__(self) -> None:
        super().__init__()

    @overrides
    def send(self, arch: ArchaiModel, dataset: DatasetProvider, budget: Optional[float] = None) -> None:
        return MagicMock()

    @overrides
    def fetch_all(self) -> List[Optional[float]]:
        return list()


def test_model_evaluator():
    arch = ArchaiModel(arch=MagicMock(), archid="test_archid", metadata={})
    dataset = MagicMock()

    # Assert that mocked value is returned
    model_evaluator = MyModelEvaluator()
    value = model_evaluator.evaluate(arch, dataset, budget=None)
    assert value == 0.0


def test_async_model_evaluator():
    arch = ArchaiModel(arch=MagicMock(), archid="test_archid", metadata={})
    dataset = MagicMock()

    # Assert that mocked method runs
    async_model_evaluator = MyAsyncModelEvaluator()
    assert async_model_evaluator.send(arch, dataset, budget=None)

    # Assert that mocked value is returned
    values = async_model_evaluator.fetch_all()
    assert isinstance(values, list)
