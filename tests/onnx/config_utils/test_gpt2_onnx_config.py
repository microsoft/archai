# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch
from transformers import PretrainedConfig

from archai.nlp.onnx.config_utils.gpt2_onnx_config import (
    GPT2FlexOnnxConfig,
    GPT2OnnxConfig,
)


@pytest.fixture
def dummy_config_gpt2():
    class DummyConfig(PretrainedConfig):
        max_position_embeddings = 16
        hidden_size = 32
        n_layer = 3
        num_attention_heads = 4

    return DummyConfig()


@pytest.fixture
def dummy_config_gpt2_flex():
    class DummyConfig(PretrainedConfig):
        max_position_embeddings = 16
        hidden_size = 32
        n_layer = 3
        num_attention_heads = [4, 4, 4]

    return DummyConfig()


def test_gpt2_onnx_config(dummy_config_gpt2):
    # Assert that default values are set correctly
    gpt2_onnx_config = GPT2OnnxConfig(dummy_config_gpt2)
    assert gpt2_onnx_config.is_ort_graph_optimizable is True
    assert gpt2_onnx_config.ort_graph_optimizer_args == (4, 32)


def test_gpt2_flex_onnx_config(dummy_config_gpt2_flex):
    # Assert that dummy inputs are generated correctly
    onnx_config = GPT2FlexOnnxConfig(dummy_config_gpt2_flex, use_past=True)
    inputs = onnx_config.generate_dummy_inputs(batch_size=3, seq_len=4, past_seq_len=2)
    assert torch.equal(inputs["input_ids"], torch.zeros((3, 4), dtype=torch.long))
    assert torch.equal(inputs["past_key_values"][0], torch.zeros((2, 3, 4, 2, 8)))
    assert torch.equal(inputs["past_key_values"][1], torch.zeros((2, 3, 4, 2, 8)))
    assert torch.equal(inputs["past_key_values"][2], torch.zeros((2, 3, 4, 2, 8)))
