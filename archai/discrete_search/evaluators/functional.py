# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Optional

from overrides import overrides

from archai.api.dataset_provider import DatasetProvider
from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.model_evaluator import ModelEvaluator


class EvaluationFunction(ModelEvaluator):
    """Custom function evaluator.

    This evaluator is used to wrap a custom evaluation function.

    """

    def __init__(self, evaluation_fn: Callable) -> None:
        """Initialize the evaluator.

        Args:
            evaluation_fn: Evaluation function that receives the parameters
                (model: ArchaiModel, dataloader: torch.utils.data.Dataloader, budget: float) and outputs
                a float.
        """

        self.evaluation_fn = evaluation_fn

    @overrides
    def evaluate(self, model: ArchaiModel, budget: Optional[float] = None) -> float:
        return self.evaluation_fn(model, budget)
