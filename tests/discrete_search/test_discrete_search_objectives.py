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
from archai.discrete_search.search_spaces.segmentation_dag.search_space import (
    SegmentationDagSearchSpace,
)


@pytest.fixture
def search_space():
    return SegmentationDagSearchSpace(nb_classes=1, img_size=(96, 96), seed=42)


@pytest.fixture
def models(search_space):
    return [search_space.random_sample() for _ in range(10)]


@pytest.fixture
def sample_input() -> Tuple[torch.Tensor]:
    return (torch.randn(1, 3, 96, 96),)


# Disables tests that use SegmentationDagSearchSpace for now because they fail in
# GPU due to tensorwatch.model_graph.torchstat.analyzer (L46) hard coding inputs to torch.long
# def test_eval_all_objs(sample_input, models):
#     search_objectives = SearchObjectives(cache_objective_evaluation=False)

#     search_objectives.add_objective(
#         "Number of parameters",
#         TorchNumParameters(),
#         higher_is_better=False,
#         compute_intensive=False,
#         constraint=(0.0, 5e5),
#     )

#     search_objectives.add_objective(
#         "OnnxLatency", AvgOnnxLatency(input_shape=(1, 3, 96, 96), num_trials=3), higher_is_better=False
#     )

#     search_objectives.add_objective("Budget Value", EvaluationFunction(lambda m, d, b: b), higher_is_better=True)

#     result = search_objectives.eval_all_objs(
#         models, None, budgets={"Budget Value": list(range(len(models)))}, progress_bar=True
#     )

#     assert all(len(r) == len(models) for r in result.values())


# def test_eval_subsets(sample_input, models):
#     # Precalculates number of params
#     num_params_obj = TorchNumParameters()
#     num_params = [num_params_obj.evaluate(m, None, None) for m in models]
#     max_params = max(num_params)

#     search_objectives = SearchObjectives(cache_objective_evaluation=False)

#     search_objectives.add_objective(
#         "Flops",
#         TorchFlops(sample_args=sample_input),
#         higher_is_better=False,
#         compute_intensive=False,
#         constraint=(0.0, float("inf")),
#     )

#     search_objectives.add_objective(
#         "OnnxLatency", AvgOnnxLatency(input_shape=(1, 3, 96, 96), num_trials=3), higher_is_better=False
#     )

#     search_objectives.add_constraint("NumParameters", TorchNumParameters(), (max_params - 0.5, max_params + 0.5))

#     search_objectives.add_objective("Budget Value", EvaluationFunction(lambda m, d, b: b), higher_is_better=True)

#     result = search_objectives.eval_cheap_objs(
#         models, None, budgets={"Budget Value": list(range(len(models)))}, progress_bar=True
#     )

#     assert set(result.keys()) == {"Flops"}

#     c_values, c_indices = search_objectives.validate_constraints(models, None)

#     assert len(c_values) == 2
#     assert len(c_indices) == 1

#     result = search_objectives.eval_expensive_objs(
#         models, None, budgets={"Budget Value": list(range(len(models)))}, progress_bar=True
#     )

#     assert set(result.keys()) == {"OnnxLatency", "Budget Value"}


# def test_eval_cache(sample_input, models):
#     so = SearchObjectives(cache_objective_evaluation=True)

#     so.add_objective(
#         "Flops",
#         TorchFlops(sample_args=sample_input),
#         higher_is_better=False,
#         compute_intensive=False,
#         constraint=(0.0, float("inf")),
#     )

#     so.add_objective("OnnxLatency", AvgOnnxLatency(input_shape=(1, 3, 96, 96), num_trials=3), higher_is_better=False)

#     so.add_constraint("NumberOfParameters", TorchNumParameters(), (0, float("inf")))

#     so.add_constraint("Random number", EvaluationFunction(lambda m, d, b: random.random()), (0.0, 1.0))

#     result = so.eval_cheap_objs(models, None, progress_bar=True)

#     assert len(result) == 1
#     assert ("Flops", models[0].archid, "NoneType", None) in so.cache

#     assert so.is_model_valid(models[0], None)
#     assert ("NumberOfParameters", models[0].archid, "NoneType", None) in so.cache
#     assert ("Random number", models[0].archid, "NoneType", None) in so.cache

#     cached_val = so.cache[("Random number", models[0].archid, "NoneType", None)]
#     cons_vals, cons_filtered = so.validate_constraints(models, None, False)

#     assert len(cons_filtered) == len(models)
#     assert cons_vals["Random number"][0] == cached_val
