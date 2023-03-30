# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from typing import Tuple

import pytest
import torch

from archai.discrete_search.api.search_objectives import SearchObjectives
from archai.discrete_search.evaluators.functional import EvaluationFunction
from archai.discrete_search.evaluators.onnx_model import AvgOnnxLatency
from archai.discrete_search.evaluators.pt_profiler import TorchFlops, TorchNumParameters
from archai.discrete_search.search_spaces.nlp.transformer_flex.search_space import (
    TransformerFlexSearchSpace,
)


@pytest.fixture
def search_space():
    return TransformerFlexSearchSpace("gpt2")


@pytest.fixture
def models(search_space):
    return [search_space.random_sample() for _ in range(3)]


@pytest.fixture
def sample_input() -> Tuple[torch.Tensor]:
    return torch.zeros(1, 1, 192, dtype=torch.long)


def test_eval_all_objs(models):
    search_objectives = SearchObjectives(cache_objective_evaluation=False)
    search_objectives.add_objective(
        "Number of parameters",
        TorchNumParameters(),
        higher_is_better=False,
        compute_intensive=False,
        constraint=(0.0, 5e5),
    )
    search_objectives.add_objective(
        "OnnxLatency",
        AvgOnnxLatency(input_shape=(1, 1, 192), num_trials=1, input_dtype="torch.LongTensor"),
        higher_is_better=False,
    )
    search_objectives.add_objective("Budget Value", EvaluationFunction(lambda m, b: b), higher_is_better=True)

    # Assert that objectives are evaluated and return a dictionary
    result = search_objectives.eval_all_objs(
        models, budgets={"Budget Value": list(range(len(models)))}, progress_bar=True
    )
    assert all(len(r) == len(models) for r in result.values())


def test_eval_subsets(sample_input, models):
    num_params_obj = TorchNumParameters()
    num_params = [num_params_obj.evaluate(m) for m in models]
    max_params = max(num_params)

    search_objectives = SearchObjectives(cache_objective_evaluation=False)
    search_objectives.add_objective(
        "Flops",
        TorchFlops(forward_args=sample_input),
        higher_is_better=False,
        compute_intensive=False,
        constraint=(0.0, float("inf")),
    )
    search_objectives.add_objective(
        "OnnxLatency",
        AvgOnnxLatency(input_shape=(1, 1, 192), num_trials=1, input_dtype="torch.LongTensor"),
        higher_is_better=False,
    )
    search_objectives.add_constraint("NumParameters", TorchNumParameters(), (max_params - 0.5, max_params + 0.5))
    search_objectives.add_objective("Budget Value", EvaluationFunction(lambda m, b: b), higher_is_better=True)

    # Assert that cheap objectives are evaluated and return a dictionary
    result = search_objectives.eval_cheap_objs(
        models, budgets={"Budget Value": list(range(len(models)))}, progress_bar=True
    )
    assert set(result.keys()) == {"Flops"}

    # Assert that constraints are valid
    c_values, c_indices = search_objectives.validate_constraints(models)
    assert len(c_values) == 2
    assert len(c_indices) == 1

    # Assert that expensive objectives are evaluated and return a dictionary
    result = search_objectives.eval_expensive_objs(
        models, budgets={"Budget Value": list(range(len(models)))}, progress_bar=True
    )
    assert set(result.keys()) == {"OnnxLatency", "Budget Value"}


def test_eval_cache(sample_input, models):
    search_objectives = SearchObjectives(cache_objective_evaluation=True)
    search_objectives.add_objective(
        "Flops",
        TorchFlops(forward_args=sample_input),
        higher_is_better=False,
        compute_intensive=False,
        constraint=(0.0, float("inf")),
    )
    search_objectives.add_objective(
        "OnnxLatency",
        AvgOnnxLatency(input_shape=(1, 1, 192), num_trials=1, input_dtype="torch.LongTensor"),
        higher_is_better=False,
    )
    search_objectives.add_constraint("NumberOfParameters", TorchNumParameters(), (0, float("inf")))
    search_objectives.add_constraint("Random number", EvaluationFunction(lambda m, b: random.random()), (0.0, 1.0))

    # Assert that cheap objectives are evaluated and cached
    result = search_objectives.eval_cheap_objs(models, progress_bar=True)
    assert len(result) == 1
    assert search_objectives.lookup_cache("Flops", models[0].archid, None)

    assert search_objectives.is_model_valid(models[0])
    assert search_objectives.lookup_cache("NumberOfParameters", models[0].archid, None)
    assert search_objectives.lookup_cache("Random number", models[0].archid, None)

    # Assert that cached value is correct and constraints are valid
    cached_val = search_objectives.lookup_cache("Random number", models[0].archid, None)
    cons_vals, cons_filtered = search_objectives.validate_constraints(models, False)
    assert cons_vals["Random number"][0] == cached_val
    assert len(cons_filtered) == len(models)
