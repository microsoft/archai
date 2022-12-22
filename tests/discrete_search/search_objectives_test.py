import pytest
import random

from archai.discrete_search.api.search_objectives import SearchObjectives
from archai.discrete_search.search_spaces.segmentation_dag import SegmentationDagSearchSpace

from archai.discrete_search.evaluators.onnx_model import AvgOnnxLatency
from archai.discrete_search.evaluators.torch_model import TorchFlops, TorchNumParameters
from archai.discrete_search.evaluators.functional import EvaluationFunction


@pytest.fixture
def search_space():
    return SegmentationDagSearchSpace(nb_classes=1, img_size=(96, 96), seed=42)


@pytest.fixture
def models(search_space):
    return [search_space.random_sample() for _ in range(30)]


def test_eval_all_objs(models):
    search_objectives = SearchObjectives(cache_objective_evaluation=False)
    
    search_objectives.add_cheap_objective(
        'Number of parameters', TorchNumParameters(input_shape=(1, 3, 96, 96)), 
        higher_is_better=False, constraint=(0.0, 5e5)
    )

    search_objectives.add_expensive_objective(
        'OnnxLatency', AvgOnnxLatency(input_shape=(1, 3, 96, 96), num_trials=3), 
        higher_is_better=False
    )

    search_objectives.add_expensive_objective(
        'Budget Value', EvaluationFunction(lambda m, d, b: b),
        higher_is_better=True
    )

    result = search_objectives.eval_all_objs(models, None, budgets={
        'Budget Value': list(range(len(models)))
    }, progress_bar=True)

    assert all(len(r) == len(models) for r in result.values())


def test_eval_subsets(models):
    # Precalculates number of params
    num_params_obj = TorchNumParameters(input_shape=(1, 3, 96, 96))
    num_params = [num_params_obj.evaluate(m, None, None) for m in models]
    max_params = max(num_params)

    search_objectives = SearchObjectives(cache_objective_evaluation=False)
    
    search_objectives.add_cheap_objective(
        'Flops', TorchFlops(input_shape=(1, 3, 96, 96)), 
        higher_is_better=False, constraint=(0.0, float('inf'))
    )

    search_objectives.add_expensive_objective(
        'OnnxLatency', AvgOnnxLatency(input_shape=(1, 3, 96, 96), num_trials=3), 
        higher_is_better=False
    )

    search_objectives.add_extra_constraint(
        'NumParameters', 
        TorchNumParameters(input_shape=(1, 3, 96, 96)), 
        (max_params - .5, max_params + .5)
    )

    search_objectives.add_expensive_objective(
        'Budget Value', EvaluationFunction(lambda m, d, b: b),
        higher_is_better=True
    )

    result = search_objectives.eval_cheap_objs(
        models, None, budgets={
        'Budget Value': list(range(len(models)))
    }, progress_bar=True)

    assert set(result.keys()) == {'Flops'}

    c_values, c_indices = search_objectives.eval_constraints(
        models, None
    )

    assert len(c_values) == 2
    assert len(c_indices) == 1

    result = search_objectives.eval_expensive_objs(
        models, None, budgets={
        'Budget Value': list(range(len(models)))
    }, progress_bar=True)
    
    assert set(result.keys()) == {'OnnxLatency', 'Budget Value'}


def test_eval_cache(models):
    so = SearchObjectives(cache_objective_evaluation=True)
    
    so.add_cheap_objective(
        'Flops', TorchFlops(input_shape=(1, 3, 96, 96)), 
        higher_is_better=False,
        constraint=(0.0, float('inf'))
    )

    so.add_expensive_objective(
        'OnnxLatency', AvgOnnxLatency(input_shape=(1, 3, 96, 96), num_trials=3), 
        higher_is_better=False
    )

    so.add_extra_constraint(
        'NumberOfParameters', TorchNumParameters(input_shape=(1, 3, 96, 96)),
        (0, float('inf'))
    )

    so.add_extra_constraint(
        'Random number', 
        EvaluationFunction(lambda m, d, b: random.random()), 
        (0.0, 1.0)
    )

    result = so.eval_cheap_objs(
        models, None, progress_bar=True
    )

    assert len(result) == 1
    assert ('Flops', models[0].archid, 'NoneType', None) in so.cache

    assert so.check_model_valid(models[0], None)
    assert ('NumberOfParameters', models[0].archid, 'NoneType', None) in so.cache
    assert ('Random number', models[0].archid, 'NoneType', None) in so.cache

    cached_val = so.cache[('Random number', models[0].archid, 'NoneType', None)]
    cons_vals, cons_filtered =  so.eval_constraints(models, None, False)

    assert len(cons_filtered) == len(models)
    assert cons_vals['Random number'][0] == cached_val
