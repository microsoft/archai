# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch
from pipeline import (
    DYNAMIC_QAT_MODULE_MAP,
    ONNX_DYNAMIC_QAT_MODULE_MAP,
    prepare_with_qat,
)


# Set up a dummy model with a single linear layer
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)


def test_prepare_with_qat():
    model = DummyModel()

    # Test normal QAT preparation
    prepared_model = prepare_with_qat(model)
    assert isinstance(prepared_model.linear, DYNAMIC_QAT_MODULE_MAP[torch.nn.Linear])

    # Test ONNX-compatible QAT preparation
    prepared_model = prepare_with_qat(model, onnx_compatible=True)
    assert isinstance(prepared_model.linear, ONNX_DYNAMIC_QAT_MODULE_MAP[torch.nn.Linear])
