# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import pytest
from onnxruntime import InferenceSession
from transformers import GPT2Config, GPT2LMHeadModel

from archai.onnx.export import export_to_onnx
from archai.onnx.onnx_loader import load_from_onnx
from archai.onnx.optimization import optimize_onnx


@pytest.fixture
def onnx_model_path():
    model = GPT2LMHeadModel(config=GPT2Config(vocab_size=1, n_layer=1))
    onnx_model_path = "temp_model.onnx"

    onnx_config = export_to_onnx(model, onnx_model_path)
    ort_model_path = optimize_onnx(onnx_model_path, onnx_config)
    yield ort_model_path

    os.remove(onnx_model_path)
    os.remove(ort_model_path)


def test_load_from_onnx(onnx_model_path):
    # Assert that the `session` can be loaded from the `onnx_model_path`
    session = load_from_onnx(onnx_model_path)
    assert isinstance(session, InferenceSession)
