# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Parameters-related objectives.
"""

from typing import List, Optional

from overrides import overrides
from torch import nn

from archai.discrete_search import ArchaiModel, DatasetProvider, Objective


class TotalParamsProxy(Objective):
    """Implements a total parameters objective."""

    higher_is_better: bool = True

    def __init__(self, trainable_only: Optional[bool] = True) -> None:
        """Calculates the number of total parameters of a torch model.

        Args:
            trainable_only: Only counts parameters with `param.requires_grad`. Defaults to `True`.

        """

        self.trainable_only = trainable_only

    @overrides
    def evaluate(self, model: ArchaiModel, dataset: DatasetProvider, budget: Optional[float] = None) -> float:
        total_params = sum(
            param.numel() for param in model.arch.parameters() if not self.trainable_only or param.requires_grad
        )

        return total_params


class NonEmbeddingParamsProxy(Objective):
    """Implements a non-embedding parameters objective."""

    higher_is_better: bool = True

    def __init__(self, exclude_cls: Optional[List[nn.Module]] = None, trainable_only: Optional[bool] = True) -> None:
        """Calculates the number of non-embedding parameters of a torch model.

        Args:
            exclude_cls: List of `nn.Module` classes to be ignored during parameter counting.
            trainable_only: Only counts parameters with `param.requires_grad`. Defaults to `True`.

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
