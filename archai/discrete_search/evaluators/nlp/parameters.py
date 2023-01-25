# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Optional

from overrides import overrides
from torch import nn

from archai.api.archai_model import ArchaiModel
from archai.api.dataset_provider import DatasetProvider
from archai.api.model_evaluator import ModelEvaluator


class TotalParamsProxy(ModelEvaluator):
    """Total number of parameters."""

    def __init__(self, trainable_only: Optional[bool] = True) -> None:
        """Initializes the evaluator.

        Args:
            trainable_only: Whether only trainable parameters should be counted.

        """

        self.trainable_only = trainable_only

    @overrides
    def evaluate(self, model: ArchaiModel, dataset: DatasetProvider, budget: Optional[float] = None) -> float:
        total_params = sum(
            param.numel() for param in model.arch.parameters() if not self.trainable_only or param.requires_grad
        )

        return total_params


class NonEmbeddingParamsProxy(ModelEvaluator):
    """Total number of non-embedding parameters."""

    def __init__(self, exclude_cls: Optional[List[nn.Module]] = None, trainable_only: Optional[bool] = True) -> None:
        """Initialize the evaluator.

        Used as a proxy for the perplexity of decoder-only transformer LMs.

        Args:
            exclude_cls: List of PyTorch module classes to exclude from parameter counting.
                If `None`, defaults to `[torch.nn.Embedding]`.
            trainable_only: Whether only trainable parameters should be counted.

        Reference:
            "LiteTransformerSearch: Training-free Neural Architecture Search for
                Efficient Language Models", Javaheripi et. al, 2022

        """

        self.exclude_cls = [nn.Embedding] or exclude_cls
        self.trainable_only = trainable_only

    @overrides
    def evaluate(self, model: ArchaiModel, dataset: DatasetProvider, budget: Optional[float] = None) -> float:
        total_params = sum(
            param.numel() for param in model.arch.parameters() if not self.trainable_only or param.requires_grad
        )

        embed_params = sum(
            sum(param.numel() for param in module.parameters())
            for module in model.arch.modules()
            if isinstance(module, tuple(self.exclude_cls))
        )

        return total_params - embed_params
