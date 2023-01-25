# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from archai.common.file_utils import create_file_name_identifier
from archai.onnx.export import export_to_onnx
from archai.onnx.optimization import optimize_onnx
from archai.quantization.ptq import (
    dynamic_quantization_onnx,
    dynamic_quantization_torch,
)


@pytest.fixture
def onnx_model_path():
    model = GPT2LMHeadModel(config=GPT2Config(vocab_size=1, n_layer=1))
    onnx_model_path = "temp_model.onnx"

    onnx_config = export_to_onnx(model, onnx_model_path)
    ort_model_path = optimize_onnx(onnx_model_path, onnx_config)
    yield ort_model_path

    os.remove(onnx_model_path)
    os.remove(ort_model_path)


@pytest.fixture
def model():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 20)
            self.fc2 = torch.nn.Linear(20, 30)
            self.word_emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=20)
            self.transformer = torch.nn.ModuleDict(
                {
                    "wpe": torch.nn.Embedding(num_embeddings=10, embedding_dim=20),
                    "wte": torch.nn.Embedding(num_embeddings=10, embedding_dim=20),
                }
            )

        def forward(self, x):
            return x

    return DummyModel()


def test_dynamic_quantization_onnx(onnx_model_path):
    # Assert that the quantized model exists
    qnt_model_path = dynamic_quantization_onnx(onnx_model_path)
    assert qnt_model_path == create_file_name_identifier(onnx_model_path, "-int8")
    assert os.path.exists(qnt_model_path)

    os.remove(qnt_model_path)


def test_dynamic_quantization_torch(model):
    # Assert that the quantized model has the expected properties
    model_qnt = dynamic_quantization_torch(model)
    assert isinstance(model_qnt, torch.nn.Module)
    assert isinstance(model_qnt.fc1, torch.nn.quantized.Linear)
    assert isinstance(model_qnt.fc2, torch.nn.quantized.Linear)
    assert isinstance(model_qnt.word_emb, torch.nn.quantized.Embedding)
    assert isinstance(model_qnt.transformer["wpe"], torch.nn.quantized.Embedding)
    assert isinstance(model_qnt.transformer["wte"], torch.nn.quantized.Embedding)
