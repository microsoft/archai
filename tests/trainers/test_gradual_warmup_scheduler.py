# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import pytest
import torch
from torch.optim import SGD

from archai.trainers.gradual_warmup_scheduler import GradualWarmupScheduler


@pytest.fixture
def optimizer():
    return SGD([torch.randn(2, 2, requires_grad=True)], 0.1)


@pytest.fixture
def scheduler(optimizer):
    return GradualWarmupScheduler(optimizer, 2.0, 5)


def test_gradual_warmup_scheduler(scheduler):
    # Assert that attributes have been defined correctly
    assert scheduler.multiplier == 2.0
    assert scheduler.total_epoch == 5
    assert scheduler.after_scheduler is None
    assert scheduler.finished is False

    # Assert that it produces corret values for last_epoch <= total_epoch
    scheduler.last_epoch = 3
    scheduler.base_lrs = [0.1, 0.2, 0.3]
    result = scheduler.get_lr()
    result = [np.round(lr, 2) for lr in result]
    assert result == [0.16, 0.32, 0.48]

    # Assert that it produces corret values for last_epoch > total_epoch
    scheduler.last_epoch = 7
    result = scheduler.get_lr()
    assert result == [0.2, 0.4, 0.6]

    # Assert that it produces corret values for last_epoch <= total_epoch
    scheduler.last_epoch = 3
    scheduler.base_lrs = [0.1, 0.2, 0.3]
    scheduler.step()
    result = scheduler.optimizer.param_groups[0]["lr"]
    assert np.round(result, 2) == 0.18

    # Assert that it produces corret values for last_epoch > total_epoch
    scheduler.last_epoch = 7
    scheduler.step()
    result = scheduler.optimizer.param_groups[0]["lr"]
    assert result == 0.2
