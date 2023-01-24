# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import transformers

from archai.quantization.quantizers import FakeDynamicQuant


class FakeDynamicQuantHFConv1D(transformers.modeling_utils.Conv1D):
    """Translate a huggingface/transformers Conv1D layer into a QAT-ready Conv1D layer."""

    _FLOAT_MODULE = transformers.modeling_utils.Conv1D

    def __init__(
        self,
        *args,
        dynamic_weight: Optional[bool] = True,
        activation_reduce_range: Optional[bool] = True,
        bits: Optional[int] = 8,
        onnx_compatible: Optional[bool] = False,
        qconfig: Optional[Dict[torch.nn.Module, Any]] = None,
        **kwargs,
    ) -> None:
        """Initialize a fake quantized Conv1D layer.

        Args:
            dynamic_weight: Whether to use dynamic weights.
            activation_reduce_range: Whether to reduce the range of activations.
            bits: Number of quantization bits.
            onnx_compatible: Whether quantization is compatible with ONNX.
            qconfig: Quantization configuration.

        """

        super().__init__(*args, **kwargs)

        self.dynamic_weight = dynamic_weight
        if dynamic_weight:
            self.weight_fake_quant = FakeDynamicQuant(
                dtype=torch.qint8,
                reduce_range=False,
                bits=bits,
                onnx_compatible=onnx_compatible,
            )

        self.input_pre_process = FakeDynamicQuant(
            reduce_range=activation_reduce_range,
            bits=bits,
            onnx_compatible=onnx_compatible,
        )

    @property
    def fake_quant_weight(self) -> torch.Tensor:
        """Return a fake quantization over the weight matrix."""

        return self.weight_fake_quant(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass over the fake quantized Conv1D layer.

        Args:
            x: Input tensor.

        Returns:
            Quantized tensor.

        """

        x = self.input_pre_process(x)
        size_out = x.size()[:-1] + (self.nf,)

        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.fake_quant_weight)
        x = x.view(*size_out)

        return x

    @classmethod
    def from_float(
        cls: FakeDynamicQuantHFConv1D,
        mod: torch.nn.Module,
        qconfig: Optional[Dict[torch.nn.Module, Any]] = None,
        activation_reduce_range: Optional[bool] = True,
        **kwargs,
    ) -> FakeDynamicQuantHFConv1D:
        """Map module from float to QAT-ready.

        Args:
            mod: Module to be mapped.
            qconfig: Quantization configuration.
            activation_reduce_range: Whether to reduce the range of activations.

        Returns:
            QAT-ready module.

        """

        assert type(mod) == cls._FLOAT_MODULE, (
            " qat." + cls.__name__ + ".from_float only works for " + cls._FLOAT_MODULE.__name__
        )

        if not qconfig:
            assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
            assert mod.qconfig, "Input float module must have a valid qconfig"
            qconfig = mod.qconfig

        qat_conv1d = cls(
            mod.nf,
            mod.weight.shape[0],
            activation_reduce_range=activation_reduce_range,
            qconfig=qconfig,
            **kwargs,
        )

        qat_conv1d.weight = mod.weight
        qat_conv1d.bias = mod.bias

        return qat_conv1d

    def to_float(self) -> torch.nn.Module:
        """Map module from QAT-ready to float.

        Returns:
            Float-based module.

        """

        weight = self.weight_fake_quant(self.weight)

        float_conv1d = transformers.modeling_utils.Conv1D(self.nf, self.weight.shape[0])

        float_conv1d.weight = torch.nn.Parameter(weight)
        float_conv1d.bias = self.bias

        return float_conv1d


class FakeDynamicQuantHFConv1DForOnnx(FakeDynamicQuantHFConv1D):
    """Allow a QAT-ready huggingface/transformers Conv1D layer to be exported with ONNX."""

    def __init__(self, *args, **kwargs):
        """Initialize a fake quantized Conv1D layer compatible with ONNX."""

        kwargs["activation_reduce_range"] = False
        kwargs["onnx_compatible"] = True

        super().__init__(*args, **kwargs)
