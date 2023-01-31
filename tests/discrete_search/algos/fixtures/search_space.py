# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from random import Random
from typing import List

import numpy as np
import pytest
from overrides import overrides

from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.search_space import (
    BayesOptSearchSpace,
    EvolutionarySearchSpace,
)


class DummySearchSpace(EvolutionarySearchSpace, BayesOptSearchSpace):
    def __init__(self, seed: int = 10) -> None:
        self.rng = Random(seed)

    @overrides
    def random_sample(self) -> ArchaiModel:
        return ArchaiModel(None, archid=str(self.rng.randint(0, 100_000)))

    @overrides
    def mutate(self, arch: ArchaiModel) -> ArchaiModel:
        archid = arch.archid
        return ArchaiModel(None, str(int(archid) + self.rng.randint(-10, 10)))

    @overrides
    def crossover(self, arch_list: List[ArchaiModel]) -> ArchaiModel:
        m1, m2 = arch_list[:2]
        new_archid = int((int(m1.archid) + int(m2.archid)) / 2)
        return ArchaiModel(None, str(new_archid))

    @overrides
    def save_arch(self, model: ArchaiModel, path: str) -> None:
        json.dump({"archid": model.archid}, open(path, "w"))

    @overrides
    def load_arch(self, path: str) -> ArchaiModel:
        return ArchaiModel(None, json.load(open(path))["archid"])

    @overrides
    def load_model_weights(self, model: ArchaiModel, path: str) -> None:
        pass

    @overrides
    def save_model_weights(self, model: ArchaiModel, path: str) -> None:
        pass

    @overrides
    def encode(self, arch: ArchaiModel) -> np.ndarray:
        return np.array([int(arch.archid)])


@pytest.fixture
def search_space():
    return DummySearchSpace()
