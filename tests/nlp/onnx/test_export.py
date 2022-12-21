# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from transformers import GPT2Config, GPT2LMHeadModel

from archai.nlp.onnx.config_utils.onnx_config_base import OnnxConfig
from archai.nlp.onnx.export import export_to_onnx


def test_export_to_onnx():
    model = GPT2LMHeadModel(config=GPT2Config(vocab_size=1, n_layer=1))
    onnx_model_path = "temp_model.onnx"

    # Assert that the `onnx_config` is returned
    onnx_config = export_to_onnx(model, onnx_model_path)
    assert isinstance(onnx_config, OnnxConfig)

    # Assert that the `onnx_config` is returned when `use_past` is set to `False`
    onnx_config = export_to_onnx(model, onnx_model_path, use_past=False)
    assert isinstance(onnx_config, OnnxConfig)

    # Assert that the `onnx_config` is returned when `share_weights` is set to `False`
    onnx_config = export_to_onnx(model, onnx_model_path, share_weights=False)
    assert isinstance(onnx_config, OnnxConfig)

    os.remove(onnx_model_path)
