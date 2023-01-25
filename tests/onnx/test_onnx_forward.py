# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from archai.onnx.onnx_forward import gpt2_onnx_forward


def test_gpt2_onnx_forward():
    # Assert that the forward method returns the expected keys
    model = GPT2LMHeadModel(config=GPT2Config(vocab_size=128, n_layer=3))
    input_ids = torch.zeros((1, 4), dtype=torch.long)
    outputs_dict = gpt2_onnx_forward(model, input_ids)
    assert "logits" in outputs_dict.keys()
    assert "past_key_values" in outputs_dict.keys()

    # Assert that the forward method returns the expected keys
    model = GPT2LMHeadModel(config=GPT2Config(vocab_size=128, n_layer=3, use_cache=False))
    input_ids = torch.zeros((1, 4), dtype=torch.long)
    outputs_dict = gpt2_onnx_forward(model, input_ids)
    assert "logits" in outputs_dict.keys()
    assert "past_key_values" not in outputs_dict.keys()
