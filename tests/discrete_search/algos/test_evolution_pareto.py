# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from typing import Optional
from random import Random
import pytest
from overrides import overrides
from archai.discrete_search.algos.evolution_pareto import EvolutionParetoSearch
from archai.discrete_search.api.search_objectives import SearchObjectives
from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.model_evaluator import ModelEvaluator
from archai.discrete_search.search_spaces.config import (
    ArchParamTree, ConfigSearchSpace, DiscreteChoice,
)


class DummyEvaluator(ModelEvaluator):
    def __init__(self, rng: Random):
        self.dummy = True
        self.rng = rng

    @overrides
    def evaluate(self, model: ArchaiModel, budget: Optional[float] = None) -> float:
        return self.rng.random()


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("out")

@pytest.fixture
def tree_c2():
    c = {
        'p1': DiscreteChoice(list([False, True])),
        'p2': DiscreteChoice(list([False, True]))
    }

    return c


def test_evolution_pareto(output_dir, search_space, search_objectives):
    cache = []
    for _ in range(2):
        algo = EvolutionParetoSearch(search_space, search_objectives, output_dir, num_iters=3, init_num_models=5, seed=42)
        search_space.rng = algo.rng

        search_results = algo.search()
        assert len(os.listdir(output_dir)) > 0

        df = search_results.get_search_state_df()
        assert all(0 <= x <= 0.4 for x in df["Random1"].tolist())

        all_models = [m for iter_r in search_results.results for m in iter_r["models"]]

        # Checks if all registered models satisfy constraints
        _, valid_models = search_objectives.validate_constraints(all_models)
        assert len(valid_models) == len(all_models)

        cache += [[m.archid for m in all_models]]

    # make sure the archid's returned are repeatable so that search jobs can be restartable.
    assert cache[0] == cache[1]


def test_evolution_pareto_tree_search(output_dir, tree_c2):
    tree = ArchParamTree(tree_c2)

    def use_arch(c):
        if c.pick('p1'):
            return

        if c.pick('p2'):
            return

    seed = 42

    cache = []
    for _ in range(2):
        search_objectives = SearchObjectives()
        search_objectives.add_objective(
            'Dummy',
            DummyEvaluator(Random(seed)),
            higher_is_better=False,
            compute_intensive=False)
        search_space = ConfigSearchSpace(use_arch, tree, seed=seed)
        algo = EvolutionParetoSearch(search_space, search_objectives, output_dir, num_iters=3, init_num_models=5, seed=seed, save_pareto_model_weights=False)

        search_results = algo.search()
        assert len(os.listdir(output_dir)) > 0

        all_models = [m for iter_r in search_results.results for m in iter_r["models"]]

        cache += [[m.archid for m in all_models]]

    # make sure the archid's returned are repeatable so that search jobs can be restartable.
    assert cache[0] == cache[1]
