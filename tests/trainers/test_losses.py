# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from archai.trainers.losses import SmoothCrossEntropyLoss


def test_smooth_cross_entropy_loss():
    inputs = torch.randn(3, 5)
    targets = torch.tensor([1, 2, 3])

    # Assert that the loss is reduced correctly (mean)
    loss_fn = SmoothCrossEntropyLoss(reduction="mean")
    loss = loss_fn(inputs, targets)
    assert loss.shape == torch.Size([])

    # Assert that the loss is reduced correctly (sum)
    loss_fn = SmoothCrossEntropyLoss(reduction="sum")
    loss = loss_fn(inputs, targets)
    assert loss.shape == torch.Size([])

    # Assert that the loss is weighted correctly
    weight = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    loss_fn = SmoothCrossEntropyLoss(weight=weight, reduction="mean")
    loss = loss_fn(inputs, targets)
    assert loss.shape == torch.Size([])

    # Assert that the loss is smoothed correctly
    smoothing = 0.1
    loss_fn = SmoothCrossEntropyLoss(smoothing=smoothing, reduction="mean")
    loss = loss_fn(inputs, targets)
    assert loss.shape == torch.Size([])
