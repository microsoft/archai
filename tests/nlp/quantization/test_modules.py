# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch
import transformers

from archai.nlp.quantization.modules import (
    FakeDynamicQuant,
    FakeDynamicQuantConv1d,
    FakeDynamicQuantHFConv1D,
    FakeDynamicQuantLinear,
    FakeQuantEmbedding,
)


@pytest.fixture
def fake_quant_embedding():
    return FakeQuantEmbedding(num_embeddings=5, embedding_dim=3)


@pytest.fixture
def fake_dynamic_quant_linear():
    return FakeDynamicQuantLinear(in_features=3, out_features=2)


@pytest.fixture
def fake_dynamic_quant_conv1d():
    return FakeDynamicQuantConv1d(in_channels=3, out_channels=2, kernel_size=3)


@pytest.fixture
def fake_dynamic_quant_hf_conv1d():
    return FakeDynamicQuantHFConv1D(nf=3, nx=2)


def test_fake_quant_embedding_init(fake_quant_embedding):
    # Assert that the `fake_quant_embedding` is initialized correctly
    assert fake_quant_embedding.num_embeddings == 5
    assert fake_quant_embedding.embedding_dim == 3
    assert isinstance(fake_quant_embedding.weight_fake_quant, FakeDynamicQuant)


def test_fake_quant_embedding_fake_quant_weight(fake_quant_embedding):
    # Assert that the `fake_quant_weight` has correct shape and type
    fake_quant_weight = fake_quant_embedding.fake_quant_weight
    assert fake_quant_weight.shape == (5, 3)
    assert isinstance(fake_quant_weight, torch.Tensor)


def test_fake_quant_embedding_forward(fake_quant_embedding):
    x = torch.tensor([0, 1, 2, 3, 4])

    # Assert that the `output` has correct shape and type
    output = fake_quant_embedding(x)
    assert output.shape == (5, 3)
    assert isinstance(output, torch.Tensor)


def test_fake_quant_embedding_from_float():
    mod = torch.nn.Embedding(num_embeddings=5, embedding_dim=3)
    qconfig = {}

    # Assert that the `quantized_mod` has correct attributes, values and types
    quantized_mod = FakeQuantEmbedding.from_float(mod, qconfig)
    assert quantized_mod.num_embeddings == mod.num_embeddings
    assert quantized_mod.embedding_dim == mod.embedding_dim
    assert quantized_mod.weight.model_parallel is False


def test_fake_quant_embedding_to_float(fake_quant_embedding):
    # Assert that the `float_mod` has correct attributes, values and types
    float_mod = fake_quant_embedding.to_float()
    assert float_mod.num_embeddings == fake_quant_embedding.num_embeddings
    assert float_mod.embedding_dim == fake_quant_embedding.embedding_dim
    assert float_mod.weight.model_parallel is True


def test_fake_dynamic_quant_linear_init(fake_dynamic_quant_linear):
    # Assert that the `fake_dynamic_quant_linear` is initialized correctly
    assert fake_dynamic_quant_linear.in_features == 3
    assert fake_dynamic_quant_linear.out_features == 2
    assert isinstance(fake_dynamic_quant_linear.weight_fake_quant, FakeDynamicQuant)
    assert isinstance(fake_dynamic_quant_linear.input_pre_process, FakeDynamicQuant)


def test_fake_dynamic_quant_linear_fake_quant_weight(fake_dynamic_quant_linear):
    # Assert that the `fake_quant_weight` has correct shape and type
    fake_quant_weight = fake_dynamic_quant_linear.fake_quant_weight
    assert fake_quant_weight.shape == (2, 3)
    assert isinstance(fake_quant_weight, torch.Tensor)


def test_fake_dynamic_quant_linear_forward(fake_dynamic_quant_linear):
    x = torch.randn(4, 3)

    # Assert that the `output` has correct shape and type
    output = fake_dynamic_quant_linear(x)
    assert output.shape == (4, 2)
    assert isinstance(output, torch.Tensor)


def test_fake_dynamic_quant_linear_from_float():
    mod = torch.nn.Linear(in_features=3, out_features=2)
    qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")

    # Assert that the `quantized_mod` has correct attributes, values and types
    quantized_mod = FakeDynamicQuantLinear.from_float(mod, qconfig)
    assert quantized_mod.in_features == mod.in_features
    assert quantized_mod.out_features == mod.out_features
    assert torch.equal(quantized_mod.weight, mod.weight)
    assert torch.equal(quantized_mod.bias, mod.bias)
    assert isinstance(quantized_mod.weight_fake_quant, FakeDynamicQuant)
    assert isinstance(quantized_mod.input_pre_process, FakeDynamicQuant)


def test_fake_dynamic_quant_linear_to_float(fake_dynamic_quant_linear):
    # Assert that the `float_mod` has correct attributes, values and types
    float_mod = fake_dynamic_quant_linear.to_float()
    assert float_mod.in_features == fake_dynamic_quant_linear.in_features
    assert float_mod.out_features == fake_dynamic_quant_linear.out_features
    assert torch.equal(float_mod.weight, fake_dynamic_quant_linear.weight_fake_quant(fake_dynamic_quant_linear.weight))
    assert torch.equal(float_mod.bias, fake_dynamic_quant_linear.bias)


def test_fake_dynamic_quant_conv1d_init(fake_dynamic_quant_conv1d):
    # Assert that the `fake_dynamic_quant_conv1d` is initialized correctly
    assert fake_dynamic_quant_conv1d.in_channels == 3
    assert fake_dynamic_quant_conv1d.out_channels == 2
    assert fake_dynamic_quant_conv1d.kernel_size == (3,)
    assert isinstance(fake_dynamic_quant_conv1d.weight_fake_quant, FakeDynamicQuant)
    assert isinstance(fake_dynamic_quant_conv1d.input_pre_process, FakeDynamicQuant)


def test_fake_dynamic_quant_conv1d_fake_quant_weight(fake_dynamic_quant_conv1d):
    # Assert that the `fake_quant_weight` has correct shape and type
    fake_quant_weight = fake_dynamic_quant_conv1d.fake_quant_weight
    assert fake_quant_weight.shape == (2, 3, 3)
    assert isinstance(fake_quant_weight, torch.Tensor)


def test_fake_dynamic_quant_conv1d_forward(fake_dynamic_quant_conv1d):
    x = torch.randn(3, 3)

    # Assert that the `output` has correct shape and type
    output = fake_dynamic_quant_conv1d(x)
    assert output.shape == (2, 1)
    assert isinstance(output, torch.Tensor)


def test_fake_dynamic_quant_conv1d_from_float():
    mod = torch.nn.Conv1d(in_channels=3, out_channels=2, kernel_size=3)
    qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")

    # Assert that the `quantized_mod` has correct attributes, values and types
    quantized_mod = FakeDynamicQuantConv1d.from_float(mod, qconfig)
    assert quantized_mod.in_channels == mod.in_channels
    assert quantized_mod.out_channels == mod.out_channels
    assert quantized_mod.kernel_size == mod.kernel_size
    assert torch.equal(quantized_mod.weight, mod.weight)
    assert torch.equal(quantized_mod.bias, mod.bias)
    assert isinstance(quantized_mod.weight_fake_quant, FakeDynamicQuant)
    assert isinstance(quantized_mod.input_pre_process, FakeDynamicQuant)


def test_fake_dynamic_quant_conv1d_to_float(fake_dynamic_quant_conv1d):
    # Assert that the `float_mod` has correct attributes, values and types
    float_mod = fake_dynamic_quant_conv1d.to_float()
    assert float_mod.in_channels == fake_dynamic_quant_conv1d.in_channels
    assert float_mod.out_channels == fake_dynamic_quant_conv1d.out_channels
    assert float_mod.kernel_size == fake_dynamic_quant_conv1d.kernel_size
    assert torch.equal(float_mod.weight, fake_dynamic_quant_conv1d.weight_fake_quant(fake_dynamic_quant_conv1d.weight))
    assert torch.equal(float_mod.bias, fake_dynamic_quant_conv1d.bias)


def test_fake_dynamic_quant_hf_conv1d_init(fake_dynamic_quant_hf_conv1d):
    # Assert that the `fake_dynamic_quant_hf_conv1d` is initialized correctly
    assert fake_dynamic_quant_hf_conv1d.nf == 3
    assert isinstance(fake_dynamic_quant_hf_conv1d.weight_fake_quant, FakeDynamicQuant)
    assert isinstance(fake_dynamic_quant_hf_conv1d.input_pre_process, FakeDynamicQuant)


def test_fake_dynamic_quant_hf_conv1d_fake_quant_weight(fake_dynamic_quant_hf_conv1d):
    # Assert that the `fake_quant_weight` has correct shape and type
    fake_quant_weight = fake_dynamic_quant_hf_conv1d.fake_quant_weight
    assert fake_quant_weight.shape == (2, 3)
    assert isinstance(fake_quant_weight, torch.Tensor)


def test_fake_dynamic_quant_hf_conv1d_forward(fake_dynamic_quant_hf_conv1d):
    x = torch.randn(3, 2)

    # Assert that the `output` has correct shape and type
    output = fake_dynamic_quant_hf_conv1d(x)
    assert output.shape == (3, 3)
    assert isinstance(output, torch.Tensor)


def test_fake_dynamic_quant_hf_conv1d_from_float():
    mod = transformers.modeling_utils.Conv1D(nf=3, nx=2)
    qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")

    # Assert that the `quantized_mod` has correct attributes, values and types
    quantized_mod = FakeDynamicQuantHFConv1D.from_float(mod, qconfig)
    assert quantized_mod.nf == mod.nf
    assert torch.equal(quantized_mod.weight, mod.weight)
    assert torch.equal(quantized_mod.bias, mod.bias)
    assert isinstance(quantized_mod.weight_fake_quant, FakeDynamicQuant)
    assert isinstance(quantized_mod.input_pre_process, FakeDynamicQuant)


def test_fake_dynamic_quant_hf_conv1d_to_float(fake_dynamic_quant_hf_conv1d):
    # Assert that the `float_mod` has correct attributes, values and types
    float_mod = fake_dynamic_quant_hf_conv1d.to_float()
    assert float_mod.nf == fake_dynamic_quant_hf_conv1d.nf
    assert torch.equal(
        float_mod.weight, fake_dynamic_quant_hf_conv1d.weight_fake_quant(fake_dynamic_quant_hf_conv1d.weight)
    )
    assert torch.equal(float_mod.bias, fake_dynamic_quant_hf_conv1d.bias)
