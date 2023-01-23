# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import List, Optional

from overrides import EnforceOverrides

from archai.api.archai_model import ArchaiModel
from archai.api.dataset_provider import DatasetProvider


class ModelEvaluator(EnforceOverrides):
    """Abstract class for synchronous model evaluators.

    Evaluators are general-use classes used to evaluate architectures in
    given criteria (task performance, speed, size, etc.).

    Subclasses of `ModelEvaluator` are expected to implement `ModelEvaluator.evaluate`.

    Synchronous evaluators are computed by search algorithms sequentially.
    For parallel / async. execution, please refer to `archai.api.AsyncModelEvaluator`.

    For a list of bult-in evaluators, please check `archai.discrete_search.evaluators`.

    Examples:
        >>> class MyValTaskAccuracy(ModelEvaluator):
        >>>     def __init__(self, batch_size: int = 32):
        >>>         self.batch_size = batch_size
        >>>
        >>>     @overrides
        >>>     def evaluate(self, model: ArchaiModel, dataset: DatasetProvider, budget: Optional[float] = None):
        >>>         _, val_data = dataset.get_train_val_datasets()
        >>>         val_dl = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size)
        >>>
        >>>         with torch.no_grad():
        >>>             labels = np.concatenate([y for _, y in val_dl], axis=0)
        >>>             preds = np.concatenate(
        >>>                 [model.arch(x).cpu().numpy() for x, _ in val_dl],
        >>>                 axis=0
        >>>             )
        >>>
        >>>         return np.mean(labels == preds)

        >>> class NumberOfModules(ModelEvaluator):
        >>>     @overrides
        >>>     def evaluate(self, model: ArchaiModel, dataset: DatasetProvider, budget: Optional[float] = None):
        >>>         return len(list(model.arch.modules()))

    """

    @abstractmethod
    def evaluate(self, arch: ArchaiModel, dataset: DatasetProvider, budget: Optional[float] = None) -> float:
        """Evaluates an `ArchaiModel` instance, optionally using a `DatasetProvider` and budget value.

        Args:
            arch: Model to be evaluated.
            dataset: A dataset provider object.
            budget: A budget multiplier value, used by search algorithms like `SucessiveHalving`
                to specify how much compute should be spent in this evaluation. In order to use
                this type of search algorithm, the implementation of `evaluate()` must use the
                passed `budget` value accordingly.

        Returns:
            Evaluation result.

        """

        pass


class AsyncModelEvaluator(EnforceOverrides):
    """Abstract class for asynchronous model evaluators.

    Evaluators are general-use classes used to evaluate architectures in given criteria
    (task performance, speed, size, etc.).

    Unlike `archai.api.ModelEvaluator`, `AsyncModelEvaluator` evaluates models in asynchronous
    fashion, by sending evaluation jobs to a queue and fetching the results later.

    Subclasses of `AsyncModelEvaluator` are expected to implement
    `AsyncModelEvaluator.send(arch: ArchaiModel, dataset: DatasetProvider, budget: Optional[float])`
    and `AsyncModelEvaluator.fetch_all()`.

    `AsyncModelEvaluator.send` is a non-blocking call that schedules an evaluation job for a
    given (model, dataset, budget) triplet. `AsyncModelEvaluator.fetch_all` is a blocking call
    that waits and gathers the results from current evaluation jobs and cleans the job queue.

    For a list of bult-in evaluators, please check `archai.discrete_search.evaluators`.

    >>> my_obj = MyAsyncObj()  # My AsyncModelEvaluator subclass
    >>>
    >>> # Non blocking calls
    >>> my_obj.send(model_1, dataset_provider, budget=None)
    >>> my_obj.send(model_2, dataset_provider, budget=None)
    >>> my_obj.send(model_3, dataset_provider, budget=None)
    >>>
    >>> # Blocking call
    >>> eval_results = my_obj.fetch_all()
    >>> assert len(eval_resuls) == 3
    >>>
    >>> # Job queue is reset after `fetch_call` method
    >>> my_obj.send(model_4, dataset_provider, budget=None)
    >>> assert len(my_obj.fetch_all()) == 1

    """

    @abstractmethod
    def send(self, arch: ArchaiModel, dataset: DatasetProvider, budget: Optional[float] = None) -> None:
        """Sends an evaluation job for a given (model, dataset, budget) triplet.

        Args:
            arch: Model to be evaluated
            dataset: A dataset provider object
            budget: A budget multiplier value, used by search algorithms like `SucessiveHalving`
                to specify how much compute should be spent in this evaluation. In order to use
                this type of search algorithm, the implementation of `send()` must use the passed
                `budget` value accordingly.

        """

        pass

    @abstractmethod
    def fetch_all(self) -> List[Optional[float]]:
        """Fetches all evaluation results from the job queue.

        Returns:
            List of evaluation results. Each result is a `float` `None` if evaluation job failed.

        """

        pass
