# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch

from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.search_objectives import SearchObjectives
from archai.discrete_search.api.search_results import SearchResults
from archai.discrete_search.evaluators.pt_profiler import TorchNumParameters
from archai.discrete_search.search_spaces.nlp.transformer_flex.search_space import (
    TransformerFlexSearchSpace,
)


def test_search_results():
    search_space = TransformerFlexSearchSpace("gpt2")
    objectives = SearchObjectives()
    search_results = SearchResults(search_space, objectives)

    # Assert that attributes are set correctly
    assert search_results.search_space == search_space
    assert search_results.objectives == objectives
    assert search_results.iteration_num == 0
    assert len(search_results.search_walltimes) == 0
    assert len(search_results.results) == 0


def test_add_iteration_results():
    search_space = TransformerFlexSearchSpace("gpt2")

    objectives = SearchObjectives()
    objectives.add_objective("n_parameters", TorchNumParameters(), False)

    search_results = SearchResults(search_space, objectives)
    models = [ArchaiModel(torch.nn.Linear(10, 1), "archid")]

    obj_name = objectives.objective_names[0]
    evaluation_results = {obj_name: np.array([0.5], dtype=np.float32)}
    search_results.add_iteration_results(models, evaluation_results)

    # Assert that attributes are set correctly after calling `add_iteration_results`
    assert search_results.iteration_num == 1
    assert len(search_results.search_walltimes) == 1
    assert len(search_results.results) == 1
    assert len(search_results.results[0]["models"]) == 1
    assert search_results.results[0][obj_name][0] == 0.5
