# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List
from unittest.mock import MagicMock

import numpy as np
from overrides import overrides

from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.search_space import (
    BayesOptSearchSpace,
    DiscreteSearchSpace,
    EvolutionarySearchSpace,
)


class MyDiscreteSearchSpace(DiscreteSearchSpace):
    def __init__(self) -> None:
        super().__init__()

    @overrides
    def save_arch(self, arch: ArchaiModel, file_path: str) -> None:
        return MagicMock()

    @overrides
    def load_arch(self, file_path: str) -> ArchaiModel:
        return MagicMock()

    @overrides
    def save_model_weights(self, arch: ArchaiModel, file_path: str) -> None:
        return MagicMock()

    @overrides
    def load_model_weights(self, arch: ArchaiModel, file_path: str) -> None:
        return MagicMock()

    @overrides
    def random_sample(self) -> ArchaiModel:
        return MagicMock()


class MyEvolutionarySearchSpace(MyDiscreteSearchSpace, EvolutionarySearchSpace):
    def __init__(self) -> None:
        super().__init__()

    @overrides
    def mutate(self, arch: ArchaiModel) -> ArchaiModel:
        return MagicMock()

    @overrides
    def crossover(self, arch_list: List[ArchaiModel]) -> ArchaiModel:
        return MagicMock()


class MyBayesOptSearchSpace(MyDiscreteSearchSpace, BayesOptSearchSpace):
    def __init__(self) -> None:
        super().__init__()

    @overrides
    def encode(self, arch: ArchaiModel) -> np.ndarray:
        return MagicMock()


def test_discrete_search_space():
    search_space = MyDiscreteSearchSpace()

    # Assert that overridden methods run
    assert search_space.save_arch(MagicMock(), "test")
    assert search_space.load_arch("test")
    assert search_space.save_model_weights(MagicMock(), "test")
    assert search_space.load_model_weights(MagicMock(), "test")
    assert search_space.random_sample()


def test_evolutionary_search_space():
    search_space = MyEvolutionarySearchSpace()

    # Assert that overridden methods run
    assert search_space.mutate(MagicMock())
    assert search_space.crossover([MagicMock(), MagicMock()])


def test_bayes_opt_search_space():
    search_space = MyBayesOptSearchSpace()

    # Assert that overridden methods run
    assert search_space.encode(MagicMock())
