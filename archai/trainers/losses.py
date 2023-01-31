# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


class SmoothCrossEntropyLoss(_WeightedLoss):
    """Cross entropy loss with label smoothing support."""

    def __init__(
        self, weight: Optional[torch.Tensor] = None, reduction: Optional[str] = "mean", smoothing: Optional[float] = 0.0
    ) -> None:
        """Initialize the loss function.

        Args:
            weight: Weight for each class.
            reduction: Reduction method.
            smoothing: Label smoothing factor.

        """

        super().__init__(weight=weight, reduction=reduction)

        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing: Optional[float] = 0.0) -> torch.Tensor:
        assert 0 <= smoothing < 1
        with torch.no_grad():
            # For label smoothing, we replace 1-hot vector with 0.9-hot vector instead.
            # Create empty vector of same size as targets, fill them up with smoothing/(n-1)
            # then replace element where 1 supposed to go and put there 1-smoothing instead
            targets = (
                torch.empty(size=(targets.size(0), n_classes), device=targets.device)
                .fill_(smoothing / (n_classes - 1))
                .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
            )
        return targets

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1), self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:  # To support weighted targets
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss
