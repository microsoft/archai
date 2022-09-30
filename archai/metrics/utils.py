from typing import Union, List, Dict

import numpy as np
import warnings
from dataclasses import dataclass
from tqdm import tqdm

from archai.nas.arch_meta import ArchWithMetaData
from archai.datasets.dataset_provider import DatasetProvider
from archai.metrics.base import BaseMetric, BaseAsyncMetric


def evaluate_models(models: List[ArchWithMetaData],
                    objectives: Dict[str, Union[BaseMetric, BaseAsyncMetric]],  
                    dataset_providers: Union[DatasetProvider, List[DatasetProvider]],
                    budgets: Union[Dict[str, float], Dict[str, List[float]], None] = None) -> Dict[str, np.ndarray]:
    """Evaluates all objective functions on a list of models and dataset(s).
    
    Metrics are evaluated in the following order:
        (1) Asynchronous metrics are dispatched by calling `.send`
        (2) Synchronous metrics are computed using `.compute`
        (3) Asynchronous metrics results are gathered by calling `.fetch_all`

    Args:
        models (List[ArchWithMetadata]): List of architectures from a search space.
        
        objectives (Mapping[str, Union[BaseMetric, BaseAsyncMetric]]): Dictionary mapping
            an objective identifier to a metric (either `BaseMetric` or `BaseAsyncMetric`), e.g:
                ```
                   {
                        'Latency (ms)': MyMetricX(),
                        'Validation Accuracy': MyMetricY(),
                        ...
                   } 
                ```.
        
        dataset_providers (Union[DatasetProvider, List[DatasetProvider]]): A single dataset provider
             or list of dataset providers with the same length of `models`.
        
        budgets (Union[Dict[str, float], Dict[str, List[float]], None]): Dictionary containing budget values
            for each objective or objective-model combination:
            1. Constant budget for each objective function (Dict[str, float]), e.g:
                ```
                    {
                        'Latency (ms)': 1.0,
                        'Validation Accuracy': 2.5
                    }
                ```
            2. Budget value for each objective-model combination (Dict[str, List[float]]), e.g:
                ```
                    {
                        'Latency (ms)': [
                            1.0, # Budget for model 1
                            2.0, # Budget for model 2
                            ...
                        ],
                        'Validation Accuracy': [
                            ...
                        ]

                    }
                ```
            3. Default budget for all objectives (`budgets=None`)
            Defaults to None.
    
    Returns:
        Dict[str, np.array]: Evaluation results (`np.array` of size `len(models)`) for each metric passed
            in `objectives`.
    """

    assert all(isinstance(obj, (BaseMetric, BaseAsyncMetric)) for obj in objectives.values()),\
        'All objectives must subclass `BaseMetric` or `BaseAsyncMetric`.'
    assert isinstance(models, list)

    if isinstance(dataset_providers, list):
        assert len(dataset_providers) == len(models)
    else:
        dataset_providers = [dataset_providers] * len(models)

    if budgets:
        for obj_name, budget in budgets.items():
            if not isinstance(budget, list):
                budgets[obj_name] = [budget] * len(models)

    objective_results = dict()
    inputs = list(enumerate(zip(models, dataset_providers)))

    sync_objectives = [t for t in objectives.items() if isinstance(t[1], BaseMetric)]
    async_objectives = [t for t in objectives.items() if isinstance(t[1], BaseAsyncMetric)]

    # Dispatches jobs for all async objectives first
    for obj_name, obj in async_objectives:
        pbar = tqdm(inputs, desc=f'Dispatching jobs for {str(obj.__class__)}...')
        
        for arch_idx, (arch, dataset) in pbar:
            if budgets:
                obj.send(arch, dataset, budget=budgets[obj_name][arch_idx])
            else:
                obj.send(arch, dataset)
    
    # Calculates synchronous objectives in order
    for obj_name, obj in sync_objectives:
        pbar = tqdm(inputs, desc=f'Calculating {str(obj.__class__)}...')

        if budgets:
            objective_results[obj_name] = np.array([
                obj.compute(arch, dataset, budget=budgets[obj_name][arch_idx]) 
                for arch_idx, (arch, dataset) in pbar
            ], dtype=np.float64)
        else:
            objective_results[obj_name] = np.array([
                obj.compute(arch, dataset) 
                for _, (arch, dataset) in pbar
            ], dtype=np.float64)

    # Gets results from async objectives
    pbar = tqdm(async_objectives, desc=f'Gathering results for async objectives...')
    for obj_name, obj in pbar:
        objective_results[obj_name] = np.array(obj.fetch_all(), dtype=np.float64)

    return objective_results


def get_pareto_frontier(models: List[ArchWithMetaData], 
                        evaluation_results: Dict[str, np.ndarray],
                        objectives: Dict[str, Union[BaseMetric, BaseAsyncMetric]]) -> Dict:
    assert len(objectives) == len(evaluation_results)
    assert all(len(r) == len(models) for r in evaluation_results.values())

    # Inverts maximization objectives 
    inverted_results = {
        obj_name: (-obj_results if objectives[obj_name].higher_is_better else obj_results)
        for obj_name, obj_results in evaluation_results.items()
    }

    # Converts results to an array of shape (len(models), len(objectives))
    results_array = np.vstack(list(inverted_results.values())).T

    pareto_points = np.array(
        _find_pareto_frontier_points(results_array, is_decreasing=True)
    )

    return {
        'models': [models[idx] for idx in pareto_points],
        'evaluation_results': {
            obj_name: obj_results[pareto_points]
            for obj_name, obj_results in evaluation_results.items()
        },
        'indices': pareto_points
    }


def _find_pareto_frontier_points(all_points: np.ndarray, is_decreasing: bool = True) -> List[int]:
    """Takes in a list of n-dimensional points, one per row, returns the list of row indices
        which are Pareto-frontier points.
        
    Assumes that lower values on every dimension are better.

    Args:
        all_points: N-dimensional points.
        is_decreasing: Whether Pareto-frontier decreases or not.

    Returns:
        List of Pareto-frontier indexes.
    """

    # For each point see if there exists  any other point which dominates it on all dimensions
    # If that is true, then it is not a pareto point and vice-versa

    # Inputs should alwyas be a two-dimensional array
    assert len(all_points.shape) == 2

    pareto_inds = []

    dim = all_points.shape[1]

    # Gets the indices of unique points
    _, unique_indices = np.unique(all_points, axis=0, return_index=True)

    for i in unique_indices:
        this_point = all_points[i,:]
        is_pareto = True

        for j in unique_indices:
            if j == i:
                continue

            other_point = all_points[j,:]

            if is_decreasing:
                diff = this_point - other_point
            else:
                diff = other_point - this_point

            if sum(diff >= 0) == dim:
                # Other point is smaller/larger on all dimensions
                # so we have found at least one dominating point
                is_pareto = False
                break
                
        if is_pareto:
            pareto_inds.append(i)

    return pareto_inds


def _find_non_dominated_sorting(all_points: np.ndarray, is_decreasing: bool = True) -> List[List[int]]:
    """Finds non-dominated sorting frontiers from a matrix (#points, #objectives).

    Args:
        all_points (np.ndarray): N-dimensional points.
        is_decreasing (bool, optional): Whether pareto-frontier decreases or not. Defaults to True.

    Returns:
        List[List[int]]: List of frontier indices
    """    
    all_indices = np.arange(all_points.shape[0])
    available_indices = np.ones_like(all_indices).astype(bool)
    frontiers = []

    while available_indices.sum() > 0:
        frontier_indices = _find_pareto_frontier_points(
            all_points[available_indices], is_decreasing=is_decreasing
        )
        frontier_indices = all_indices[available_indices][frontier_indices]

        frontiers.append(frontier_indices.tolist())
        available_indices[frontier_indices] = 0

    return frontiers
