# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from transformers import GPT2Config, GPT2LMHeadModel

from archai.common.file_utils import create_file_name_identifier
from archai.onnx.export import export_to_onnx
from archai.onnx.optimization import optimize_onnx


def test_optimize_onnx():
    model = GPT2LMHeadModel(config=GPT2Config(vocab_size=1, n_layer=1))
    onnx_model_path = "temp_model.onnx"
    onnx_config = export_to_onnx(model, onnx_model_path)

    # Assert that `ort_model_path` + `-opt` is returned when `optimize_onnx` is called
    ort_model_path = optimize_onnx(onnx_model_path, onnx_config)
    assert ort_model_path == create_file_name_identifier(onnx_model_path, "-opt")

    # Assert that `ort_model_path` + `opt` is returned when `optimize_onnx`
    # is called with `only_ort` set to `True`
    ort_model_path = optimize_onnx(onnx_model_path, onnx_config, only_ort=True)
    assert ort_model_path == create_file_name_identifier(onnx_model_path, "-opt")

    # Assert that `ort_model_path` + `-opt` is returned when `optimize_onnx`
    # is called with `input_int32` set to `True`
    ort_model_path = optimize_onnx(onnx_model_path, onnx_config, input_int32=True)
    assert ort_model_path == create_file_name_identifier(onnx_model_path, "-opt")

    os.remove(onnx_model_path)
    os.remove(ort_model_path)
