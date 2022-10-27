# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
from abc import abstractmethod
from typing import List, Optional
from overrides import EnforceOverrides

from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.dataset import DatasetProvider


class Objective(EnforceOverrides):
    """Abstract base class for synchronous objective functions. Objectives are general-use classes used 
    to evaluate architectures in given criteria (task performance, speed, size, etc.).
    Objectives are optimized by search algorithms.

    Subclasses of `Objective` are expected to implement `Objective.evaluate`, that evaluates the objective,
    and override the `higher_is_better` property, that specifices the optimization direction used 
    by search algorithms.

    Synchronous objectives are computed by search algorithms sequentially. For parallel / async. execution, 
    please refer to `archai.discrete_search.AsyncObjective`.

    **Examples**

    .. highlight:: python
    .. code-block:: python
       :caption: Task Accuracy

        class MyValTaskAccuracy(Objective):
            higher_is_better: bool = True

            def __init__(self, batch_size: int = 32):
                self.batch_size = batch_size
            
            @overrides
            def evaluate(self, model: ArchaiModel, dataset: DatasetProvider,
                         budget: Optional[float] = None):
                _, val_data = dataset.get_train_val_datasets()
                val_dl = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size)
                
                with torch.no_grad():
                    labels = np.concatenate([y for _, y in val_dl], axis=0)
                    preds = np.concatenate([
                        model.arch(x).cpu().numpy() for x, _ in val_dl
                    ], axis=0)
                
                return np.mean(labels == preds)
    
    .. highlight:: python
    .. code-block:: python
       :caption: Number of modules

        class NumberOfModules(Objective):
            ''' Class that measures the size of a model by
            the number of torch modules '''
            
            higher_is_better: bool = False 

            @overrides
            def evaluate(self, model: ArchaiModel, dataset: DatasetProvider,
                         budget: Optional[float] = None):
                return len(list(model.arch.modules()))

    For a list of bultin metrics, please check `archai.discrete_search.objectives`.
    """
    higher_is_better: bool = False

    @abstractmethod
    def evaluate(self, arch: ArchaiModel, dataset: DatasetProvider,
                 budget: Optional[float] = None) -> float:
        """Evaluates a given ArchaiModel, optionally using a DatasetProvider and budget value.

        Args:
            arch (ArchaiModel): Model to be evaluated
            dataset (DatasetProvider): A dataset provider object

            budget (Optional[float], optional): A budget multiplier value, used by search algorithms like 
                `SucessiveHalving` to specify how much compute should be spent in this evaluation.
                In order to use this type of search algorithm, the implementation of `eval` must
                use the passed `budget` value accordingly.
                Defaults to None.

        Returns:
            float: Evaluation result
        """


class AsyncObjective(EnforceOverrides):
    """Abstract base class for asynchronous objective functions. Objectives are general-use classes used 
    to evaluate architectures in given criteria (task performance, speed, size, etc.).
    Objectives are optimized by search algorithms.

    Unlike `archai.discrete_search.Objective`, `AsyncObjective` defines an objective function 
    with async. / parallel execution. 
    
    Subclasses of `AsyncObjective` are expected to implement
    `AsyncObjective.send(arch: ArchaiModel, dataset: DatasetProvider, budget: Optional[float])` 
    and `AsyncObjective.fetch_all()`, and override the `higher_is_better` class attribute, that specifices
    the optimization direction used by search algorithms.

    `AsyncObjective.send` is a non-blocking call that schedules an evaluation job for a given (model, dataset, budget)
    triplet. `AsyncObjective.fetch_all` is a blocking call that waits and gathers the results from previously
    sent evaluation jobs and cleans the job queue.

    .. highlight:: python
    .. code-block:: python
        :caption: Task Accuracy

        my_obj = MyAsyncObj()  # My AsyncObjective subclass
        
        # Non blocking calls
        my_obj.send(model_1, dataset_provider, budget=None)
        my_obj.send(model_2, dataset_provider, budget=None)
        my_obj.send(model_3, dataset_provider, budget=None)

        # Blocking call
        eval_results = my_obj.fetch_all()
        assert len(eval_resuls) == 3

        # Job queue is reset after `fetch_call` method
        my_obj.send(model_4, dataset_provider, budget=None)
        assert len(my_obj.fetch_all()) == 1

    For a list of bultin objectives, please check `archai.discrete_search.objectives`.
    """

    higher_is_better: bool = False

    @abstractmethod
    def send(self, arch: ArchaiModel, dataset: DatasetProvider,
             budget: Optional[float] = None) -> None:
        """Sends an evaluation job to be computed asynchronously.

        Args:
            arch (ArchaiModel): Evaluated model
            dataset (DatasetProvider): Dataset
            
            budget (Optional[float], optional): A budget multiplier value, used by search algorithms like 
                `SucessiveHalving` to specify how much compute should be spent in this evaluation.
                Defaults to None.
            
        """


    @abstractmethod
    def fetch_all(self) -> List[Optional[float]]:
        """Fetch all the results from active jobs sent using the `.send` method and resets job queue. The
        results are expected to respect the original scheduling order.
        """
