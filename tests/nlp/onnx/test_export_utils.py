# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import types

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from archai.nlp.file_utils import calculate_onnx_model_size
from archai.nlp.onnx.export import export_to_onnx
from archai.nlp.onnx.export_utils import prepare_model_for_onnx, weight_sharing
from archai.nlp.onnx.onnx_forward import gpt2_onnx_forward


def test_prepare_model_for_onnx():
    # Assert that the forward function is replaced with `gpt2_onnx_forward`
    # and that the `c_fc` and `c_proj` layers are replaced with `torch.nn.Linear`
    model = GPT2LMHeadModel(config=GPT2Config(vocab_size=1, n_layer=1))
    model = prepare_model_for_onnx(model, model_type="gpt2")
    assert model.forward == types.MethodType(gpt2_onnx_forward, model)
    assert isinstance(model.transformer.h[0].mlp.c_fc, torch.nn.Linear)
    assert isinstance(model.transformer.h[0].mlp.c_proj, torch.nn.Linear)


def test_weight_sharing():
    model = GPT2LMHeadModel(config=GPT2Config(vocab_size=1, n_layer=1))
    onnx_model_path = "temp_model.onnx"
    export_to_onnx(model, onnx_model_path, share_weights=False)
    onnx_size = calculate_onnx_model_size(onnx_model_path)

    # Assert that the embedding and softmax weights are shared in the ONNX model
    weight_sharing(onnx_model_path, "gpt2")
    onnx_size_shared_weights = calculate_onnx_model_size(onnx_model_path)
    assert onnx_size_shared_weights < onnx_size

    os.remove(onnx_model_path)
