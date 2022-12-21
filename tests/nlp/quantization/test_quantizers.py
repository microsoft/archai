# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from archai.nlp.quantization.quantizers import FakeDynamicQuant


def test_fake_dynamic_quant():
    x = torch.randn(4)

    # Assert the quint8 quantization type with 8-bit
    fake_quant = FakeDynamicQuant(dtype=torch.quint8, bits=8)
    y = fake_quant(x)
    assert y.dtype == torch.float32
    assert torch.equal(
        y,
        torch.fake_quantize_per_tensor_affine(
            x, fake_quant._scale, fake_quant._zero_pointer, fake_quant.qmin, fake_quant.qmax
        ),
    )

    # Assert the qint8 quantization type with 8-bit
    fake_quant = FakeDynamicQuant(dtype=torch.qint8, bits=8)
    y = fake_quant(x)
    assert y.dtype == torch.float32
    assert torch.equal(
        y,
        torch.fake_quantize_per_tensor_affine(
            x, fake_quant._scale, fake_quant._zero_pointer, fake_quant.qmin, fake_quant.qmax
        ),
    )

    # Assert the quint8 quantization type with 16-bit
    fake_quant = FakeDynamicQuant(dtype=torch.quint8, bits=16)
    y = fake_quant(x)
    assert y.dtype == torch.float32
    assert torch.equal(
        y,
        torch.fake_quantize_per_tensor_affine(
            x, fake_quant._scale, fake_quant._zero_pointer, fake_quant.qmin, fake_quant.qmax
        ),
    )

    # Assert the qint8 quantization type with 16-bit
    fake_quant = FakeDynamicQuant(dtype=torch.qint8, bits=16)
    y = fake_quant(x)
    assert y.dtype == torch.float32
    assert torch.equal(
        y,
        torch.fake_quantize_per_tensor_affine(
            x, fake_quant._scale, fake_quant._zero_pointer, fake_quant.qmin, fake_quant.qmax
        ),
    )

    # Assert the `onnx_compatible` option for 8-bit
    fake_quant = FakeDynamicQuant(dtype=torch.quint8, bits=8, onnx_compatible=True)
    y = fake_quant(x)
    assert y.dtype == torch.float32
    assert torch.equal(
        y,
        torch.fake_quantize_per_tensor_affine(
            x, fake_quant._scale, fake_quant._zero_pointer, fake_quant.qmin, fake_quant.qmax
        ),
    )

    fake_quant = FakeDynamicQuant(dtype=torch.qint8, bits=8, onnx_compatible=True)
    y = fake_quant(x)
    assert y.dtype == torch.float32
    assert torch.equal(
        y,
        torch.fake_quantize_per_tensor_affine(
            x, fake_quant._scale, fake_quant._zero_pointer, fake_quant.qmin, fake_quant.qmax
        ),
    )
