# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch
from torch.optim import SGD

from archai.trainers.cyclic_cosine_scheduler import CyclicCosineDecayLR

INITIAL_LR = 1.0


@pytest.fixture
def optimizer():
    return SGD([torch.randn(2, 2, requires_grad=True)], INITIAL_LR)


@pytest.fixture
def scheduler(optimizer):
    return CyclicCosineDecayLR(optimizer, init_decay_epochs=5, min_decay_lr=0.1)


def test_cyclic_cosine_decay_lr_init(optimizer):
    # Assert for invalid init_decay_epochs input
    with pytest.raises(ValueError):
        CyclicCosineDecayLR(optimizer, init_decay_epochs=0, min_decay_lr=0.1)
    with pytest.raises(ValueError):
        CyclicCosineDecayLR(optimizer, init_decay_epochs=-1, min_decay_lr=0.1)

    # Assert for invalid min_decay_lr input
    with pytest.raises(ValueError):
        CyclicCosineDecayLR(optimizer, init_decay_epochs=5, min_decay_lr=[0.1, 0.2])

    # Assert for invalid restart_interval input
    with pytest.raises(ValueError):
        CyclicCosineDecayLR(optimizer, init_decay_epochs=5, min_decay_lr=0.1, restart_interval=0)
    with pytest.raises(ValueError):
        CyclicCosineDecayLR(optimizer, init_decay_epochs=5, min_decay_lr=0.1, restart_interval=-1)

    # Assert for invalid restart_interval_multiplier input
    with pytest.raises(ValueError):
        CyclicCosineDecayLR(optimizer, init_decay_epochs=5, min_decay_lr=0.1, restart_interval_multiplier=0)
    with pytest.raises(ValueError):
        CyclicCosineDecayLR(optimizer, init_decay_epochs=5, min_decay_lr=0.1, restart_interval_multiplier=-1)

    # Assert for invalid restart_lr input
    with pytest.raises(ValueError):
        CyclicCosineDecayLR(optimizer, init_decay_epochs=5, min_decay_lr=0.1, restart_lr=[0.1, 0.2])

    # Assert for invalid warmup_epochs input
    with pytest.raises(ValueError):
        CyclicCosineDecayLR(optimizer, init_decay_epochs=5, min_decay_lr=0.1, warmup_epochs=0)
    with pytest.raises(ValueError):
        CyclicCosineDecayLR(optimizer, init_decay_epochs=5, min_decay_lr=0.1, warmup_epochs=-1)

    # Assert for invalid warmup_start_lr input
    with pytest.raises(ValueError):
        CyclicCosineDecayLR(optimizer, init_decay_epochs=5, min_decay_lr=0.1, warmup_epochs=1)
    with pytest.raises(ValueError):
        CyclicCosineDecayLR(optimizer, init_decay_epochs=5, min_decay_lr=0.1, warmup_epochs=1, warmup_start_lr=1)
    with pytest.raises(ValueError):
        CyclicCosineDecayLR(
            optimizer, init_decay_epochs=5, min_decay_lr=0.1, warmup_epochs=1, warmup_start_lr=[0.1, 0.2]
        )


def test_cyclic_cosine_decay_lr_step(scheduler, optimizer):
    # Assert that the learning rate decreases after each step
    lrs = []
    for _ in range(15):
        scheduler.step()
        lrs.append([param_group["lr"] for param_group in optimizer.param_groups])
    assert lrs[-1] < lrs[0]

    # Assert that the learning rate restarts after the specified number of epochs
    lrs = []
    scheduler = CyclicCosineDecayLR(optimizer, init_decay_epochs=10, min_decay_lr=0.1, restart_interval=5)
    for _ in range(15):
        scheduler.step()
        lrs.append([param_group["lr"] for param_group in optimizer.param_groups])
    assert lrs[-1] == [INITIAL_LR]

    # Assert that the learning rate follows a warmup schedule
    lrs = []
    scheduler = CyclicCosineDecayLR(
        optimizer, init_decay_epochs=10, min_decay_lr=0.1, warmup_epochs=5, warmup_start_lr=0.01
    )
    for _ in range(10):
        scheduler.step()
        lrs.append([param_group["lr"] for param_group in optimizer.param_groups])
    assert lrs[-1] > lrs[0]
