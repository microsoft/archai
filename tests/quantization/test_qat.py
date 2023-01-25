# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy

import pytest
import torch

from archai.quantization.modules import FakeDynamicQuantLinear
from archai.quantization.qat import (
    DYNAMIC_QAT_MODULE_MAP,
    ONNX_DYNAMIC_QAT_MODULE_MAP,
    float_to_qat_modules,
    prepare_with_qat,
    qat_to_float_modules,
)


@pytest.fixture
def model():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

    return DummyModel()


def test_float_to_qat_modules(model):
    # Assert that the QAT linear layer is an instance of `FakeDynamicQuantLinear`
    float_to_qat_modules(model, qconfig=torch.quantization.get_default_qat_qconfig("qnnpack"))
    assert isinstance(model.linear, FakeDynamicQuantLinear)


def test_qat_to_float_modules(model):
    # Assert that the converted model linear layer is an instance of `torch.nn.Linear`
    float_to_qat_modules(model, qconfig=torch.quantization.get_default_qat_qconfig("qnnpack"))
    qat_to_float_modules(model)
    assert isinstance(model.linear, torch.nn.Linear)


def test_prepare_with_qat(model):
    # Assert normal QAT preparation
    model_copy = copy.deepcopy(model)
    prepare_with_qat(model_copy)
    assert isinstance(model_copy.linear, DYNAMIC_QAT_MODULE_MAP[torch.nn.Linear])

    # Assert normal QAT preparation without `inplace`
    prepared_model = prepare_with_qat(model, inplace=False)
    assert isinstance(prepared_model.linear, DYNAMIC_QAT_MODULE_MAP[torch.nn.Linear])

    # Assert ONNX-compatible QAT preparation
    model_copy = copy.deepcopy(model)
    prepare_with_qat(model_copy, onnx_compatible=True)
    assert isinstance(model_copy.linear, ONNX_DYNAMIC_QAT_MODULE_MAP[torch.nn.Linear])

    # Assert ONNX-compatible QAT preparation without `inplace`
    prepared_model = prepare_with_qat(model, inplace=False, onnx_compatible=True)
    assert isinstance(prepared_model.linear, ONNX_DYNAMIC_QAT_MODULE_MAP[torch.nn.Linear])
