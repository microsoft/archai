# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
from abc import abstractmethod
from typing import List, Optional
from overrides import EnforceOverrides

from archai.discrete_search.api.model import NasModel
from archai.discrete_search.api.dataset import DatasetProvider


class Metric(EnforceOverrides):
    """Abstract base class for synchronous metrics. Metrics are general-use classes used 
    to evaluate architectures in given criteria (task performance, speed, size, etc.).
    Metrics are used in Archai as objective functions, to be optimized by search algorithms.

    Subclasses of `Metric` are expected to implement `Metric.compute`, that computes the given metric,
    and override the `higher_is_better` property, that specifices the optimization direction used 
    by search algorithms.

    Synchronous metrics are computed by search algorithms sequentially. For parallel / async. execution, 
    please refer to `archai.discrete_search.AsyncMetric`.

    Examples:
        Task accuracy
        ```python

            class MyValTaskAccuracy(Metric):
                higher_is_better: bool = True

                def __init__(self, batch_size: int = 32):
                    self.batch_size = batch_size
                
                @overrides
                def compute(self, model: NasModel, dataset: DatasetProvider, budget: Optional[float] = None):
                    _, val_data = dataset.get_train_val_datasets()
                    val_dl = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size)
                    
                    with torch.no_grad():
                        labels = np.concatenate([y for _, y in val_dl], axis=0)
                        preds = np.concatenate([model.arch(x).cpu().numpy() for x, _ in val_dl], axis=0)
                    
                    return np.sum(labels == preds)
        ```

        Number of modules
        ```python

            class NumberOfModules(Metric):
                ''' Class that measures the size of a model by the number of torch modules '''
                
                higher_is_better: bool = False 

                @overrides
                def compute(self, model: NasModel, dataset: DatasetProvider, budget: Optional[float] = None):
                    return len(list(model.arch.modules()))
        ```

    For a list of bultin metrics, please check `archai.discrete_search.metrics`.
    """
    higher_is_better: bool = False

    @abstractmethod
    def compute(self, arch: NasModel, dataset: DatasetProvider,
                budget: Optional[float] = None) -> float:
        """Computes the metric for a given NasModel, DatasetProvider and optionally a budget value.

        Args:
            arch (NasModel): Model to be evaluated
            dataset (DatasetProvider): A dataset provider object
            budget (Optional[float], optional): A budget multiplier value, used by search algorithms like 
                `SucessiveHalving` specifying how much compute should be spent in this evaluation.
                Defaults to None.

        Returns:
            float: Computed metric value
        """

    def __neg__(self) -> 'Metric':
        neg_metric = copy.deepcopy(self)
        neg_metric.higher_is_better = not neg_metric.higher_is_better
        return neg_metric


class AsyncMetric(EnforceOverrides):
    """Abstract base class for asynchronous metrics. Metrics are general-use classes used 
    to evaluate architectures in given criteria (task performance, speed, size, etc.).
    Metrics are used in Archai as objective functions, to be optimized by search algorithms.

    Unlike `archai.discrete_search.AsyncMetric`, `AsyncMetric` defines a metric 
    with async. / parallel execution. 
    
    Subclasses of `AsyncMetric` are expected to implement
    `AsyncMetric.send(arch: NasModel, dataset: DatasetProvider, budget: Optional[float])` 
    and `AsyncMetric.fetch_all()`, and override the `higher_is_better` property, that specifices
    the optimization direction used by search algorithms.

    `AsyncMetric.send` is a non-blocking call that schedules a job for a given (model, dataset, budget)
    triplet. `AsyncMetric.fetch_all` is a blocking call that waits and gathers the results from previously
    sent jobs and cleans the job queue.

        ```python
        my_metric = MyAsyncMetric()
        
        # Non blocking calls
        my_metric.send(model_1, dataset_provider, budget=None)
        my_metric.send(model_2, dataset_provider, budget=None)
        my_metric.send(model_3, dataset_provider, budget=None)

        # Blocking call
        eval_results = my_metric.fetch_all()
        assert len(eval_resuls) == 3

        # Job queue is reset after `fetch_call` method
        my_metric.send(model_4, dataset_provider, budget=None)
        assert len(my_metric.fetch_all()) == 1
        ```

    For a list of bultin metrics, please check `archai.discrete_search.metrics`.
    """

    higher_is_better: bool = False

    @abstractmethod
    def send(self, arch: NasModel, dataset: DatasetProvider,
             budget: Optional[float] = None) -> None:
        """Sends an evaluation job to be computed asynchronously.

        Args:
            arch (NasModel): Evaluated model
            dataset (DatasetProvider): Dataset
            budget (Optional[float], optional): A budget multiplier value, used by search algorithms like 
                `SucessiveHalving` to specifying how much compute should be spent in this evaluation.
                Defaults to None.
        """


    @abstractmethod
    def fetch_all(self) -> List[Optional[float]]:
        """Fetch all the results from active jobs sent using the `.send` method and resets job queue. The
        results are expected to respect the original scheduling order.
        """

    def __neg__(self) -> 'AsyncMetric':
        neg_metric = copy.deepcopy(self)
        neg_metric.higher_is_better = not neg_metric.higher_is_better
        return neg_metric
