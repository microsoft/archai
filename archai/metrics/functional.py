from typing import Callable, Optional
from overrides import overrides

from archai.metrics.base import BaseMetric
from archai.nas.nas_model import NasModel
from archai.datasets.dataset_provider import DatasetProvider


class FunctionalMetric(BaseMetric):
    def __init__(self, evaluation_fn: Callable, higher_is_better: bool):
        """Uses the result of a custom function as a metric.

        Args:
            evaluation_fn (Callable): Evaluation function that receives the parameters
                (model: NasModel, dataloader: torch.utils.data.Dataloader) and outputs a float.
            higher_is_better (bool): Optimization direction. True for maximization, False for minimization.
        """        
        self.evaluation_fn = evaluation_fn
        self.higher_is_better = higher_is_better

    @overrides
    def compute(self, model: NasModel, dataset_provider: DatasetProvider,
                budget: Optional[float] = None) -> float:
        return self.evaluation_fn(model, dataset_provider, budget)
