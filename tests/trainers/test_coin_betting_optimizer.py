# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch.optim import Optimizer

from archai.trainers.coin_betting_optimizer import CocobBackprop, CocobOns


def test_cocob_backprop():
    model = torch.nn.Linear(5, 5)
    loss_fn = torch.nn.MSELoss()
    optimizer = CocobBackprop(model.parameters(), alpha=100.0, eps=1e-8)

    # Asserts that the optimizer works with a dummy forward and backward passes
    optimizer.zero_grad()
    outputs = model(torch.randn(10, 5))
    loss = loss_fn(outputs, torch.randn(10, 5))
    loss.backward()
    assert loss.shape == torch.Size([])


def test_cocob_ons():
    model = torch.nn.Linear(5, 5)
    loss_fn = torch.nn.MSELoss()
    optimizer = CocobOns(model.parameters(), eps=1e-8)

    # Asserts that the optimizer works with a dummy forward and backward passes
    optimizer.zero_grad()
    outputs = model(torch.randn(10, 5))
    loss = loss_fn(outputs, torch.randn(10, 5))
    loss.backward()
    assert loss.shape == torch.Size([])
