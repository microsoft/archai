# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
from abc import abstractmethod
from typing import List, Optional, Union
from overrides import EnforceOverrides

from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.dataset_provider import DatasetProvider


class ModelEvaluator(EnforceOverrides):
    """Abstract base class for synchronous evaluators. Evaluators are general-use classes used 
    to evaluate architectures in given criteria (task performance, speed, size, etc.).

    Subclasses of `ModelEvaluator` are expected to implement `ModelEvaluator.evaluate`.

    Synchronous evaluators are computed by search algorithms sequentially. For parallel / async. execution, 
    please refer to `archai.discrete_search.api.AsyncModelEvaluator`.

    **Examples**

    .. highlight:: python
    .. code-block:: python
       :caption: Task Accuracy

        class MyValTaskAccuracy(ModelEvaluator):
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

        class NumberOfModules(ModelEvaluator):
            ''' Class that measures the size of a model by
            the number of torch modules '''

            @overrides
            def evaluate(self, model: ArchaiModel, dataset: DatasetProvider,
                         budget: Optional[float] = None):
                return len(list(model.arch.modules()))

    For a list of bultin evaluators, please check `archai.discrete_search.evaluators`.
    """

    @abstractmethod
    def evaluate(self, arch: ArchaiModel, dataset: DatasetProvider,
                 budget: Optional[float] = None) -> float:
        """Evaluates an `ArchaiModel` instance, optionally using a `DatasetProvider` and budget value.

        Args:
            arch (ArchaiModel): Model to be evaluated
            dataset (DatasetProvider): A dataset provider object

            budget (Optional[float], optional): A budget multiplier value, used by search algorithms like 
                `SucessiveHalving` to specify how much compute should be spent in this evaluation.
                In order to use this type of search algorithm, the implementation of `.evaluate` must
                use the passed `budget` value accordingly.
                Defaults to None.

        Returns:
            float: Evaluation result
        """


class AsyncModelEvaluator(EnforceOverrides):
    """Abstract base class for asynchronous evaluators. Evaluators are general-use classes used 
    to evaluate architectures in given criteria (task performance, speed, size, etc.).

    Unlike `archai.discrete_search.api.ModelEvaluator`, `AsyncModelEvaluator` evaluates models
    in asynchronous fashion, by sending evaluation jobs to a queue and fetching the results later.
    
    Subclasses of `AsyncModelEvaluator` are expected to implement
    `AsyncModelEvaluator.send(arch: ArchaiModel, dataset: DatasetProvider, budget: Optional[float])` 
    and `AsyncModelEvaluator.fetch_all()`.

    `AsyncModelEvaluator.send` is a non-blocking call that schedules an evaluation job for a given (model, dataset, budget)
    triplet. `AsyncModelEvaluator.fetch_all` is a blocking call that waits and gathers the results from current
    evaluation jobs and cleans the job queue.

    .. highlight:: python
    .. code-block:: python
        :caption: AsyncModelEvaluator usage example

        my_obj = MyAsyncObj()  # My AsyncModelEvaluator subclass
        
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

    For a list of bultin evaluators, please check `archai.discrete_search.evaluators`.
    """

    @abstractmethod
    def send(self, arch: ArchaiModel, dataset: DatasetProvider,
             budget: Optional[float] = None) -> None:
        """Sends an evaluation job for a given (model, dataset, budget) triplet.

        Args:
            arch (ArchaiModel): Model to be evaluated
            dataset (DatasetProvider): A dataset provider object
            
            budget (Optional[float], optional): A budget multiplier value, used by search algorithms like 
                `SucessiveHalving` to specify how much compute should be spent in this evaluation.
                In order to use this type of search algorithm, the implementation of `.send` must
                use the passed `budget` value accordingly.
                Defaults to None.
            
        """

    @abstractmethod
    def fetch_all(self) -> List[Optional[float]]:
        """Fetches all evaluation results from the job queue.

        Returns:
            List[Optional[float]]: List of evaluation results. Each result is a float value or None if evaluation
                job failed.
        """
