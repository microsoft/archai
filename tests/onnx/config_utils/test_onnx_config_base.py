# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch
from transformers import PretrainedConfig

from archai.onnx.config_utils.onnx_config_base import OnnxConfig, OnnxConfigWithPast


@pytest.fixture
def dummy_config():
    class DummyConfig(PretrainedConfig):
        max_position_embeddings = 16
        hidden_size = 32
        num_layers = 3
        num_attention_heads = 4

    return DummyConfig()


def test_onnx_config(dummy_config):
    # Assert that default values are set correctly
    onnx_config = OnnxConfig(dummy_config)
    assert onnx_config.config == dummy_config
    assert onnx_config.task == "causal-lm"


def test_onnx_config_get_inputs(dummy_config):
    # Assert that input names and shapes are set correctly
    onnx_config = OnnxConfig(dummy_config)
    inputs = onnx_config.get_inputs()
    assert inputs == {"input_ids": {0: "batch_size", 1: "seq_len"}}


def test_onnx_config_get_outputs(dummy_config):
    # Assert that output names and shapes are set correctly
    onnx_config = OnnxConfig(dummy_config)
    outputs = onnx_config.get_outputs()
    assert outputs == {"probs": {0: "batch_size"}}


def test_onnx_config_generate_dummy_inputs(dummy_config):
    # Assert that dummy inputs are generated correctly
    onnx_config = OnnxConfig(dummy_config)
    inputs = onnx_config.generate_dummy_inputs(batch_size=3, seq_len=10)
    assert torch.equal(inputs["input_ids"], torch.zeros((3, 10), dtype=torch.long))


def test_onnx_config_generate_dummy_inputs_exceeds_max_position_embeddings(dummy_config):
    # Assert that dummy inputs can't exceed max_position_embeddings
    onnx_config = OnnxConfig(dummy_config)
    with pytest.raises(AssertionError):
        onnx_config.generate_dummy_inputs(batch_size=3, seq_len=20)


def test_onnx_config_unsupported_task(dummy_config):
    # Assert that unsupported tasks raise an error
    with pytest.raises(AssertionError):
        OnnxConfig(dummy_config, task="unsupported_task")


def test_onnx_config_with_past_default_values(dummy_config):
    # Assert that default values are set correctly
    onnx_config = OnnxConfigWithPast(dummy_config)
    assert onnx_config.config == dummy_config
    assert onnx_config.task == "causal-lm"
    assert onnx_config.use_past is False


def test_onnx_config_with_past_get_inputs(dummy_config):
    # Assert that input names and shapes are set correctly
    onnx_config = OnnxConfigWithPast(dummy_config, use_past=True)
    inputs = onnx_config.get_inputs()
    assert inputs == {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "past_0": {1: "batch_size", 3: "past_seq_len"},
        "past_1": {1: "batch_size", 3: "past_seq_len"},
        "past_2": {1: "batch_size", 3: "past_seq_len"},
    }


def test_onnx_config_with_past_get_outputs(dummy_config):
    # Assert that output names and shapes are set correctly
    onnx_config = OnnxConfigWithPast(dummy_config, use_past=True)
    outputs = onnx_config.get_outputs()
    assert outputs == {
        "probs": {0: "batch_size"},
        "present_0": {1: "batch_size", 3: "total_seq_len"},
        "present_1": {1: "batch_size", 3: "total_seq_len"},
        "present_2": {1: "batch_size", 3: "total_seq_len"},
    }


def test_onnx_config_with_past_generate_dummy_inputs(dummy_config):
    # Assert that dummy inputs are generated correctly
    onnx_config = OnnxConfigWithPast(dummy_config, use_past=True)
    inputs = onnx_config.generate_dummy_inputs(batch_size=3, seq_len=4, past_seq_len=2)
    assert torch.equal(inputs["input_ids"], torch.zeros((3, 4), dtype=torch.long))
    assert torch.equal(inputs["past_key_values"][0], torch.zeros((2, 3, 4, 2, 8)))
    assert torch.equal(inputs["past_key_values"][1], torch.zeros((2, 3, 4, 2, 8)))
    assert torch.equal(inputs["past_key_values"][2], torch.zeros((2, 3, 4, 2, 8)))


def test_onnx_config_with_past_generate_dummy_inputs_exceeds_max_position_embeddings(dummy_config):
    # Assert that dummy inputs can't exceed max_position_embeddings
    onnx_config = OnnxConfigWithPast(dummy_config, use_past=True)
    with pytest.raises(AssertionError):
        onnx_config.generate_dummy_inputs(batch_size=3, seq_len=10, past_seq_len=8)
