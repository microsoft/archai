# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Optional

from overrides import overrides
from torch import nn

from archai.api.dataset_provider import DatasetProvider
from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.model_evaluator import ModelEvaluator


class TotalParamsProxy(ModelEvaluator):
    def __init__(self, trainable_only: Optional[bool] = True) -> None:
        """Counts the total number of trainable parameters

        Args:
            trainable_only: A flag indicating whether only trainable parameters
                should be counted. Defaults to True.
        """
        self.trainable_only = trainable_only

    @overrides
    def evaluate(self, model: ArchaiModel, dataset: DatasetProvider, budget: Optional[float] = None) -> float:
        total_params = sum(
            param.numel() for param in model.arch.parameters() if not self.trainable_only or param.requires_grad
        )

        return total_params


class NonEmbeddingParamsProxy(ModelEvaluator):
    def __init__(self, exclude_cls: Optional[List[nn.Module]] = None, trainable_only: Optional[bool] = True) -> None:
        """Total number of non-embedding parameters.
            Used as a proxy for the perplexity of decoder-only transformer LMs.

            From: "LiteTransformerSearch: Training-free Neural Architecture Search for
                Efficient Language Models", Javaheripi et. al, 2022

        Args:
            exclude_cls (Optional[List[nn.Module]], optional): List of PyTorch module classes
                to exclude from parameter counting. If `None`, defaults to `[torch.nn.Embedding]`.
                Defaults to None.
            trainable_only (Optional[bool], optional): A flag indicating whether only trainable parameters
                should be counted. Defaults to True.
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
