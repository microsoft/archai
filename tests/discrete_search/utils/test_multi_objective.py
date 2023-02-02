# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch

from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.search_objectives import SearchObjectives
from archai.discrete_search.evaluators.functional import EvaluationFunction
from archai.discrete_search.utils.multi_objective import (
    get_non_dominated_sorting,
    get_pareto_frontier,
)


def test_get_pareto_frontier():
    models = [ArchaiModel(torch.nn.Linear(10, 1), "archid") for _ in range(5)]

    evaluation_results = {
        "obj1": np.array([1, 2, 3, 4, 5]),
        "obj2": np.array([5, 4, 3, 2, 1]),
    }

    objectives = SearchObjectives()
    objectives.add_objective("obj1", EvaluationFunction(lambda m, d, b: b), higher_is_better=True)
    objectives.add_objective("obj2", EvaluationFunction(lambda m, d, b: b), higher_is_better=False)

    result = get_pareto_frontier(models, evaluation_results, objectives)

    # Assert that the result is a list of dictionaries
    assert isinstance(result, dict)

    # Assert that each dictionary has the required keys
    assert "models" in result
    assert "evaluation_results" in result
    assert "indices" in result

    # Assert that the length of each list is the same
    assert len(result["models"]) == len(result["evaluation_results"]["obj1"]) == len(result["indices"])


def test_get_non_dominated_sorting():
    models = [ArchaiModel(torch.nn.Linear(10, 1), "archid") for _ in range(5)]

    evaluation_results = {
        "obj1": np.array([1, 2, 3, 4, 5]),
        "obj2": np.array([5, 4, 3, 2, 1]),
    }

    objectives = SearchObjectives()
    objectives.add_objective("obj1", EvaluationFunction(lambda m, d, b: b), higher_is_better=True)
    objectives.add_objective("obj2", EvaluationFunction(lambda m, d, b: b), higher_is_better=False)

    result = get_non_dominated_sorting(models, evaluation_results, objectives)

    # Assert that the result is a list of dictionaries
    assert isinstance(result, list)
    assert all(isinstance(r, dict) for r in result)

    # Assert that each dictionary has the required keys
    assert all("models" in r for r in result)
    assert all("evaluation_results" in r for r in result)
    assert all("indices" in r for r in result)

    # Assert that the length of each list is the same
    assert len(result) == 5
    assert all(len(r["models"]) == len(r["evaluation_results"]["obj1"]) == len(r["indices"]) for r in result)
