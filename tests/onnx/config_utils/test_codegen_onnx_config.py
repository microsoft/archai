# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from transformers import PretrainedConfig

from archai.onnx.config_utils.codegen_onnx_config import CodeGenOnnxConfig


@pytest.fixture
def dummy_config_codegen():
    class DummyConfig(PretrainedConfig):
        max_position_embeddings = 16
        hidden_size = 32
        n_layer = 3
        num_attention_heads = 4

    return DummyConfig()


def test_codegen_onnx_config(dummy_config_codegen):
    # Assert that default values are set correctly
    codegen_onnx_config = CodeGenOnnxConfig(dummy_config_codegen)
    assert codegen_onnx_config.num_layers == 3
    assert codegen_onnx_config.is_ort_graph_optimizable is False
    assert codegen_onnx_config.ort_graph_optimizer_args == (4, 32)
