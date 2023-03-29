# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

from archai.trainers.lamb_optimizer import JITLamb, Lamb


def test_lamb_init():
    # Assert default parameter values
    lamb = Lamb([torch.randn(10, 5)])
    assert lamb.param_groups[0]["lr"] == 1e-3
    assert lamb.param_groups[0]["betas"] == (0.9, 0.999)
    assert lamb.param_groups[0]["eps"] == 1e-6
    assert lamb.param_groups[0]["weight_decay"] == 0.0
    assert lamb.adam is False

    # Assert custom parameter values
    lamb = Lamb([torch.randn(10, 5)], lr=0.5, betas=(0.8, 0.99), eps=1e-5, weight_decay=0.1, adam=True)
    assert lamb.param_groups[0]["lr"] == 0.5
    assert lamb.param_groups[0]["betas"] == (0.8, 0.99)
    assert lamb.param_groups[0]["eps"] == 1e-5
    assert lamb.param_groups[0]["weight_decay"] == 0.1
    assert lamb.adam is True

    # Assert invalid learning rate
    with pytest.raises(ValueError):
        Lamb([torch.randn(10, 5)], lr=-0.5)

    # Assert invalid epsilon value
    with pytest.raises(ValueError):
        Lamb([torch.randn(10, 5)], eps=-1e-5)

    # Assert invalid beta parameter at index 0
    with pytest.raises(ValueError):
        Lamb([torch.randn(10, 5)], betas=(-0.5, 0.99))

    # Assert invalid beta parameter at index 1
    with pytest.raises(ValueError):
        Lamb([torch.randn(10, 5)], betas=(0.8, 1.5))


def test_lamb_step():
    # Assert with closure
    def closure():
        return torch.tensor(1.0)

    lamb = Lamb([torch.randn(10, 5)])
    loss = lamb.step(closure)
    assert loss.item() == 1.0

    # Assert without closure
    lamb = Lamb([torch.randn(10, 5)])
    loss = lamb.step()
    assert loss is None


def test_jit_lamb_init():
    # Assert default parameter values
    jit_lamb = JITLamb([torch.randn(10, 5)])
    assert jit_lamb.param_groups[0]["lr"] == 1e-3
    assert jit_lamb.param_groups[0]["betas"] == (0.9, 0.999)
    assert jit_lamb.param_groups[0]["eps"] == 1e-6
    assert jit_lamb.param_groups[0]["weight_decay"] == 0.0
    assert jit_lamb.adam is False

    # Assert custom parameter values
    jit_lamb = JITLamb([torch.randn(10, 5)], lr=0.5, betas=(0.8, 0.99), eps=1e-5, weight_decay=0.1, adam=True)
    assert jit_lamb.param_groups[0]["lr"] == 0.5
    assert jit_lamb.param_groups[0]["betas"] == (0.8, 0.99)
    assert jit_lamb.param_groups[0]["eps"] == 1e-5
    assert jit_lamb.param_groups[0]["weight_decay"] == 0.1
    assert jit_lamb.adam is True

    # Assert invalid learning rate
    with pytest.raises(ValueError):
        JITLamb([torch.randn(10, 5)], lr=-0.5)

    # Assert invalid epsilon value
    with pytest.raises(ValueError):
        JITLamb([torch.randn(10, 5)], eps=-1e-5)

    # Assert invalid beta parameter at index 0
    with pytest.raises(ValueError):
        JITLamb([torch.randn(10, 5)], betas=(-0.5, 0.99))

    # Assert invalid beta parameter at index 1
    with pytest.raises(ValueError):
        JITLamb([torch.randn(10, 5)], betas=(0.8, 1.5))


def test_jit_lamb_step():
    # Assert with closure
    def closure():
        return torch.tensor(1.0)

    jit_lamb = JITLamb([torch.randn(10, 5)])
    loss = jit_lamb.step(closure)
    assert loss.item() == 1.0

    # Assert without closure
    jit_lamb = JITLamb([torch.randn(10, 5)])
    loss = jit_lamb.step()
    assert loss is None
