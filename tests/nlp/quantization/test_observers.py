# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

from archai.nlp.quantization.observers import OnnxDynamicObserver


@pytest.fixture
def onnx_dynamic_observer():
    return OnnxDynamicObserver(dtype=torch.qint8)


def test_onnx_dynamic_observer_init(onnx_dynamic_observer):
    # Assert that initialization parameters are set correctly
    assert onnx_dynamic_observer.dtype == torch.qint8
    assert onnx_dynamic_observer.qmin == -128
    assert onnx_dynamic_observer.qmax == 127


def test_onnx_dynamic_observer_call(onnx_dynamic_observer):
    x = torch.tensor([[-1.0, 0.0, 1.0]])
    onnx_dynamic_observer(x)

    # Assert that `min_val` and `max_val` are set correctly
    assert onnx_dynamic_observer.min_val == -1.0
    assert onnx_dynamic_observer.max_val == 1.0


def test_onnx_dynamic_observer_calculate_qparams(onnx_dynamic_observer):
    x = torch.tensor([[-1.0, 0.0, 1.0]])
    onnx_dynamic_observer(x)

    # Assert that `scale` and `zero_pointer` are set correctly
    scale, zero_pointer = onnx_dynamic_observer.calculate_qparams()
    assert scale == pytest.approx(1.0 / 127)
    assert zero_pointer == 0
