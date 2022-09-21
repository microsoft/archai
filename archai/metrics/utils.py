from typing import Union, List

from tqdm import tqdm

from archai.nas.arch_meta import ArchWithMetaData
from archai.datasets.dataset_provider import DatasetProvider
from archai.metrics.base import BaseMetric, BaseAsyncMetric


def calculate_objectives(objectives: List[Union[BaseMetric, BaseAsyncMetric]], 
                         models: List[ArchWithMetaData],
                         dataset_providers: Union[DatasetProvider, List[DatasetProvider]]):
    """Calculates objectives for a list of models and dataset(s).

    Args:
        objectives (List[Union[BaseMetric, BaseAsyncMetric]]): List of objectives, possibly containing 
            asynchronous metrics. All asynchronous metrics will be dispatched before normal metrics, 
            following the original list order.
        models (List[ArchWithMetadata]): List of architectures from a search space.
        dataset_providers (Union[DatasetProvider, List[DatasetProvider]]): Dataset provider or list of
             dataset providers with the same length as `models`.
    """

    assert isinstance(objectives, list)
    assert all(isinstance(obj, (BaseMetric, BaseAsyncMetric)) for obj in objectives)
    assert isinstance(models, list)

    if isinstance(dataset_providers, list):
        assert len(dataset_providers) == len(models)
    else:
        assert isinstance(dataset_providers, DatasetProvider)
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

    return objective_results
