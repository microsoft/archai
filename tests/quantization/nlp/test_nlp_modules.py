# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch
import transformers

from archai.quantization.nlp.modules import FakeDynamicQuant, FakeDynamicQuantHFConv1D


@pytest.fixture
def fake_dynamic_quant_hf_conv1d():
    return FakeDynamicQuantHFConv1D(nf=3, nx=2)


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
