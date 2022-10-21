from typing import Callable, Optional
from overrides import overrides

from archai.discrete_search import Objective, ArchaiModel, DatasetProvider


class EvaluationFunction(Objective):
    def __init__(self, evaluation_fn: Callable, higher_is_better: bool):
        """Uses the result of a custom function as an objective function.

        Args:
            evaluation_fn (Callable): Evaluation function that receives the parameters
                (model: ArchaiModel, dataloader: torch.utils.data.Dataloader, budget: float) and outputs a float.
            higher_is_better (bool): Optimization direction. True for maximization, False for minimization.
        """        
        self.evaluation_fn = evaluation_fn
        self.higher_is_better = higher_is_better

    @overrides
    def evaluate(self, model: ArchaiModel, dataset_provider: DatasetProvider,
                budget: Optional[float] = None) -> float:
        return self.evaluation_fn(model, dataset_provider, budget)
