from typing import Dict, List, Union

import numpy as np
from tqdm import tqdm

from archai.discrete_search import (
    AsyncObjective, Objective, DatasetProvider,
    ArchaiModel
)


def evaluate_models(models: List[ArchaiModel],
                    objectives: Dict[str, Union[Objective, AsyncObjective]],  
                    dataset_providers: Union[DatasetProvider, List[DatasetProvider]],
                    budgets: Union[Dict[str, float], Dict[str, List[float]], None] = None) -> Dict[str, np.ndarray]:
    """Evaluates all objective functions on a list of models and dataset(s).
    
    Objectives are evaluated in the following order:
        (1) Asynchronous objectives are dispatched by calling `.send`
        (2) Synchronous objectives are computed using `.evaluate`
        (3) Asynchronous objectives results are gathered by calling `.fetch_all`

    Args:
        models (List[ArchaiModel]): List of architectures from a search space.
        
        objectives (Mapping[str, Union[Objective, AsyncObjective]]): Dictionary mapping
            an objective identifier to an objective object (either `Objective` or `AsyncObjective`), e.g:
                ```
                   {
                        'Latency (ms)': MyObjectiveX(),
                        'Validation Accuracy': MyObjectiveY(),
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

    assert all(isinstance(obj, (Objective, AsyncObjective)) for obj in objectives.values()),\
        'All objectives must subclass `Objective` or `AsyncObjective`.'
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

    sync_objectives = [t for t in objectives.items() if isinstance(t[1], Objective)]
    async_objectives = [t for t in objectives.items() if isinstance(t[1], AsyncObjective)]

    # Dispatches jobs for all async objectives first
    for obj_name, obj in async_objectives:
        pbar = tqdm(inputs, desc=f'Dispatching jobs for "{obj_name}"...')
        
        for arch_idx, (arch, dataset) in pbar:
            if budgets:
                obj.send(arch, dataset, budget=budgets[obj_name][arch_idx])
            else:
                obj.send(arch, dataset)
    
    # Calculates synchronous objectives in order
    for obj_name, obj in sync_objectives:
        pbar = tqdm(inputs, desc=f'Calculating "{obj_name}"...')

        if budgets:
            objective_results[obj_name] = np.array([
                obj.evaluate(arch, dataset, budget=budgets[obj_name][arch_idx]) 
                for arch_idx, (arch, dataset) in pbar
            ], dtype=np.float64)
        else:
            objective_results[obj_name] = np.array([
                obj.evaluate(arch, dataset) 
                for _, (arch, dataset) in pbar
            ], dtype=np.float64)

    # Gets results from async objectives
    pbar = tqdm(async_objectives, desc=f'Gathering results from async objectives...')
    for obj_name, obj in pbar:
        objective_results[obj_name] = np.array(obj.fetch_all(), dtype=np.float64)

    return objective_results
