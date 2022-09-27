from typing import Union, List, Optional, Dict

import numpy as np
import warnings
from dataclasses import dataclass
from tqdm import tqdm

from archai.nas.arch_meta import ArchWithMetaData
from archai.datasets.dataset_provider import DatasetProvider
from archai.metrics.base import BaseMetric, BaseAsyncMetric


def evaluate_models(models: List[ArchWithMetaData],
                    objectives: List[Union[BaseMetric, BaseAsyncMetric]],  
                    dataset_providers: Union[DatasetProvider, List[DatasetProvider]]) -> np.ndarray:
    """Evaluates a list of models using a list of objective functions.

    Args:
        models (List[ArchWithMetadata]): List of architectures from a search space.
        objectives (List[Union[BaseMetric, BaseAsyncMetric]]): List of objectives, possibly containing 
            asynchronous metrics. All asynchronous metrics will be dispatched before normal metrics, 
            following the original list order.
        dataset_providers (Union[DatasetProvider, List[DatasetProvider]]): Dataset provider or list of
             dataset providers with the same length as `models`.
    
    Returns:
        np.ndarray: `np.array` of shape (len(models), len(objectives)).
    """

    assert isinstance(objectives, list)
    assert all(isinstance(obj, (BaseMetric, BaseAsyncMetric)) for obj in objectives)
    assert isinstance(models, list)

    if isinstance(dataset_providers, list):
        assert len(dataset_providers) == len(models)
    else:
        dataset_providers = [dataset_providers] * len(models)

    objective_results = dict()
    inputs = list(zip(models, dataset_providers))

    sync_objectives = [t for t in enumerate(objectives) if isinstance(t[1], BaseMetric)]
    async_objectives = [t for t in enumerate(objectives) if isinstance(t[1], BaseAsyncMetric)]

    # Dispatches jobs for all async objectives first
    for _, obj in async_objectives:        
        for arch, dataset in tqdm(inputs, desc=f'Dispatching jobs for {str(obj.__class__)}...'):
            obj.send(arch, dataset)
    
    # Calculates synchronous objectives in order
    for obj_idx, obj in sync_objectives:
        objective_results[obj_idx] = [
            obj.compute(arch, dataset) 
            for arch, dataset in tqdm(inputs, desc=f'Calculating {str(obj.__class__)}...')
        ]

    # Gets results from async objectives
    for obj_idx, obj in tqdm(async_objectives, desc=f'Gathering results for async objectives...'):
        objective_results[obj_idx] = obj.fetch_all()

    # Returns a np.array (len(models), len(objectives)) with the results.
    # by setting dtype to float, values with `None` are automatically converted to `np.nan`
    return np.array([
        objective_results[obj_idx]
        for obj_idx in range(len(objectives))
    ], dtype=np.float64).T


def get_pareto_frontier(models: List[ArchWithMetaData], 
                        evaluation_results: np.ndarray,
                        objectives: List[Union[BaseMetric, BaseAsyncMetric]]) -> Dict:    
    # Inverts maximization objectives 
    for obj_idx, obj in enumerate(objectives):
        if obj.higher_is_better:
            evaluation_results[:, obj_idx] = -evaluation_results[:, obj_idx]

    pareto_points = np.array(
        _find_pareto_frontier_points(evaluation_results, is_decreasing=True)
    )

    return {
        'models': [models[idx] for idx in pareto_points],
        'evaluation_results': evaluation_results[pareto_points, :],
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
