# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Optional
from overrides import overrides

from archai.api.archai_model import ArchaiModel
from archai.api.dataset_provider import DatasetProvider
from archai.api.model_evaluator import ModelEvaluator


class EvaluationFunction(ModelEvaluator):
    def __init__(self, evaluation_fn: Callable):
        """Custom function evaluator. This evaluator is used to wrap a custom evaluation function.

        Args:
            evaluation_fn (Callable): Evaluation function that receives the parameters
                (model: ArchaiModel, dataloader: torch.utils.data.Dataloader, budget: float) and outputs
                a float.
        """        
        self.evaluation_fn = evaluation_fn

    @overrides
    def evaluate(self, model: ArchaiModel, dataset_provider: DatasetProvider,
                budget: Optional[float] = None) -> float:
        return self.evaluation_fn(model, dataset_provider, budget)
