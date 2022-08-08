# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""PyTorch mappings between float-based and quantization-ready modules.
"""

from typing import Any, Dict, Optional, Type

import torch
import transformers
from torch.nn import functional as F

from archai.nlp.compression.quantization.quantizers import FakeDynamicQuant


class FakeQuantEmbedding(torch.nn.Embedding):
    """Translates a torch-based Embedding layer into a QAT-ready Embedding.

    """

    def __init__(self, *args, **kwargs) -> None:
        """Initializes a fake quantized Embedding layer.

        """

        bits = kwargs.pop('bits') if 'bits' in kwargs else 8
        onnx_compatible = kwargs.pop('onnx_compatible') if 'onnx_compatible' in kwargs else False

        super().__init__(*args, **kwargs)

        self.weight_fake_quant = FakeDynamicQuant(dtype=torch.qint8,
                                                  reduce_range=False,
                                                  bits=bits,
                                                  onnx_compatible=onnx_compatible)

    @property
    def fake_quant_weight(self) -> torch.Tensor:
        """Performs a fake quantization over the weight matrix.

        Returns:
            (torch.Tensor): Fake quantized weight matrix.

        """

        return self.weight_fake_quant(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass over the fake quantized Embedding layer.

        Args:
            x: Input tensor.

        Returns:
            (torch.Tensor): Quantized tensor.

        """

        return self.fake_quant_weight[x]

    @classmethod
    def from_float(cls: Type['FakeQuantEmbedding'],
                   mod: torch.nn.Module,
                   qconfig: Optional[Dict[torch.nn.Module, Any]] = None,
                   **kwargs) -> Type['FakeQuantEmbedding']:
        """Maps module from float to QAT-ready.

        Args:
            mod: Module to be mapped.
            qconfig: Quantization configuration.

        Returns:
            (Type['FakeQuantEmbedding']): QAT-ready module.

        """

        module = cls(mod.num_embeddings, mod.embedding_dim, **kwargs)

        module.weight = mod.weight
        module.weight.model_parallel = False

        return module

    def to_float(self) -> torch.nn.Module:
        """Maps module from QAT-ready to float.

        Returns:
            (torch.nn.Module): Float-based module.

        """

        module = torch.nn.Embedding(self.num_embeddings, self.embedding_dim)

        module.weight.data = self.weight_fake_quant(self.weight.data)
        module.weight.model_parallel = True

        return module


class FakeQuantEmbeddingForOnnx(FakeQuantEmbedding):
    """Allows a QAT-ready Embedding layer to be exported with ONNX.

    """

    def __init__(self, *args, **kwargs) -> None:
        """Initializes a fake quantized Embedding layer compatible with ONNX.
        
        """
        
        kwargs['onnx_compatible'] = True

        super().__init__(*args, **kwargs)


class FakeDynamicQuantLinear(torch.nn.Linear):
    """Translates a torch-based Linear layer into a QAT-ready Linear.

    """

    _FLOAT_MODULE = torch.nn.Linear

    def __init__(self,
                 *args,
                 dynamic_weight: Optional[bool] = True,
                 activation_reduce_range: Optional[bool] = True,
                 bits: Optional[int] = 8,
                 onnx_compatible: Optional[bool] = False,
                 qconfig: Optional[Dict[torch.nn.Module, Any]] = None,
                 **kwargs) -> None:
        """Initializes a fake quantized Linear layer.

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
            self.weight_fake_quant = FakeDynamicQuant(dtype=torch.qint8,
                                                      reduce_range=False,
                                                      bits=bits,
                                                      onnx_compatible=onnx_compatible)

        self.input_pre_process = FakeDynamicQuant(reduce_range=activation_reduce_range,
                                                  bits=bits,
                                                  onnx_compatible=onnx_compatible)

    @property
    def fake_quant_weight(self) -> torch.Tensor:
        """Performs a fake quantization over the weight matrix.

        Returns:
            (torch.Tensor): Fake quantized weight matrix.

        """

        return self.weight_fake_quant(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass over the fake quantized Linear layer.

        Args:
            x: Input tensor.

        Returns:
            (torch.Tensor): Quantized tensor.

        """

        x = self.input_pre_process(x)

        return F.linear(x, self.fake_quant_weight, self.bias)

    @classmethod
    def from_float(cls: Type['FakeDynamicQuantLinear'],
                   mod: torch.nn.Module,
                   qconfig: Optional[Dict[torch.nn.Module, Any]] = None,
                   activation_reduce_range: Optional[bool] = True,
                   **kwargs) -> Type['FakeDynamicQuantLinear']:
        """Maps module from float to QAT-ready.

        Args:
            mod: Module to be mapped.
            qconfig: Quantization configuration.
            activation_reduce_range: Whether to reduce the range of activations.

        Returns:
            (Type['FakeDynamicQuantLinear']): QAT-ready module.

        """

        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + cls._FLOAT_MODULE.__name__

        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert mod.qconfig, 'Input float module must have a valid qconfig'
            qconfig = mod.qconfig

        qat_linear = cls(mod.in_features,
                         mod.out_features,
                         bias=mod.bias is not None,
                         activation_reduce_range=activation_reduce_range,
                         qconfig=qconfig,
                         **kwargs)

        qat_linear.weight = mod.weight
        qat_linear.bias = mod.bias

        return qat_linear

    def to_float(self) -> torch.nn.Module:
        """Maps module from QAT-ready to float.

        Returns:
            (torch.nn.Module): Float-based module.

        """

        weight = self.weight_fake_quant(self.weight)

        float_linear = torch.nn.Linear(self.in_features,
                                       self.out_features,
                                       bias=self.bias is not None)

        float_linear.weight = torch.nn.Parameter(weight)
        float_linear.bias = self.bias

        return float_linear


class FakeDynamicQuantLinearForOnnx(FakeDynamicQuantLinear):
    """Allows a QAT-ready Linear layer to be exported with ONNX.

    """

    def __init__(self, *args,  **kwargs) -> None:
        """Initializes a fake quantized Linear layer compatible with ONNX.
        
        """

        kwargs['activation_reduce_range'] = False
        kwargs['onnx_compatible'] = True

        super().__init__(*args, **kwargs)


class FakeDynamicQuantConv1d(torch.nn.Conv1d):
    """Translates a torch-based Conv1d layer into a QAT-ready Conv1d.

    """

    _FLOAT_MODULE = torch.nn.Conv1d

    def __init__(self,
                 *args,
                 dynamic_weight: Optional[bool] = True,
                 activation_reduce_range: Optional[bool] = True,
                 bits: Optional[int] = 8,
                 onnx_compatible: Optional[bool] = False,
                 qconfig: Optional[Dict[torch.nn.Module, Any]] = None,
                 **kwargs) -> None:
        """Initializes a fake quantized Conv1d layer.

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
            self.weight_fake_quant = FakeDynamicQuant(dtype=torch.qint8,
                                                      reduce_range=False,
                                                      bits=bits,
                                                      onnx_compatible=onnx_compatible)

        self.input_pre_process = FakeDynamicQuant(reduce_range=activation_reduce_range,
                                                  bits=bits,
                                                  onnx_compatible=onnx_compatible)

    @property
    def fake_quant_weight(self) -> torch.Tensor:
        """Performs a fake quantization over the weight matrix.

        Returns:
            (torch.Tensor): Fake quantized weight matrix.

        """

        return self.weight_fake_quant(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass over the fake quantized Conv1d layer.

        Args:
            x: Input tensor.

        Returns:
            (torch.Tensor): Quantized tensor.

        """

        x = self.input_pre_process(x)

        return self._conv_forward(x, self.fake_quant_weight, self.bias)

    @classmethod
    def from_float(cls: Type['FakeDynamicQuantConv1d'],
                   mod: torch.nn.Module,
                   qconfig: Optional[Dict[torch.nn.Module, Any]] = None,
                   activation_reduce_range: Optional[bool] = True,
                   **kwargs) -> Type['FakeDynamicQuantConv1d']:
        """Maps module from float to QAT-ready.

        Args:
            mod: Module to be mapped.
            qconfig: Quantization configuration.
            activation_reduce_range: Whether to reduce the range of activations.

        Returns:
            (Type['FakeDynamicQuantConv1d']): QAT-ready module.

        """

        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + cls._FLOAT_MODULE.__name__

        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert mod.qconfig, 'Input float module must have a valid qconfig'
            qconfig = mod.qconfig

        qat_conv1d = cls(in_channels=mod.in_channels,
                         out_channels=mod.out_channels,
                         kernel_size=mod.kernel_size,
                         stride=mod.stride,
                         padding=mod.padding,
                         dilation=mod.dilation,
                         groups=mod.groups,
                         padding_mode=mod.padding_mode,
                         bias=mod.bias is not None,
                         activation_reduce_range=activation_reduce_range,
                         qconfig=qconfig,
                         **kwargs)

        qat_conv1d.weight = mod.weight
        qat_conv1d.bias = mod.bias

        return qat_conv1d

    def to_float(self) -> torch.nn.Module:
        """Maps module from QAT-ready to float.

        Returns:
            (torch.nn.Module): Float-based module.

        """

        weight = self.weight_fake_quant(self.weight)

        float_conv1d = torch.nn.Conv1d(in_channels=self.in_channels,
                                       out_channels=self.out_channels,
                                       kernel_size=self.kernel_size,
                                       stride=self.stride,
                                       padding=self.padding,
                                       dilation=self.dilation,
                                       groups=self.groups,
                                       padding_mode=self.padding_mode,
                                       bias=self.bias is not None)

        float_conv1d.weight = torch.nn.Parameter(weight)
        float_conv1d.bias = self.bias

        return float_conv1d
        

class FakeDynamicQuantConv1dForOnnx(FakeDynamicQuantConv1d):
    """Allows a QAT-ready Conv1d layer to be exported with ONNX.

    """

    def __init__(self, *args,  **kwargs) -> None:
        """Initializes a fake quantized Conv1d layer compatible with ONNX.
        
        """

        kwargs['activation_reduce_range'] = False
        kwargs['onnx_compatible'] = True

        super().__init__(*args, **kwargs)


class FakeDynamicQuantHFConv1D(transformers.modeling_utils.Conv1D):
    """Translates a huggingface's transformers Conv1D to QAT-ready.
    """

    _FLOAT_MODULE = transformers.modeling_utils.Conv1D

    def __init__(self,
                 *args,
                 dynamic_weight: Optional[bool] = True,
                 activation_reduce_range: Optional[bool] = True,
                 bits: Optional[int] = 8,
                 onnx_compatible: Optional[bool] = False,
                 qconfig: Optional[Dict[torch.nn.Module, Any]] = None,
                 **kwargs) -> None:
        """Initializes a fake quantized Conv1d layer.

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
            self.weight_fake_quant = FakeDynamicQuant(dtype=torch.qint8,
                                                      reduce_range=False,
                                                      bits=bits,
                                                      onnx_compatible=onnx_compatible)

        self.input_pre_process = FakeDynamicQuant(reduce_range=activation_reduce_range,
                                                  bits=bits,
                                                  onnx_compatible=onnx_compatible)

    @property
    def fake_quant_weight(self) -> torch.Tensor:
        """Performs a fake quantization over the weight matrix.

        Returns:
            (torch.Tensor): Fake quantized weight matrix.

        """

        return self.weight_fake_quant(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass over the fake quantized Conv1d layer.

        Args:
            x: Input tensor.

        Returns:
            (torch.Tensor): Quantized tensor.

        """

        x = self.input_pre_process(x)
        size_out = x.size()[:-1] + (self.nf,)

        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.fake_quant_weight)
        x = x.view(*size_out)

        return x

    @classmethod
    def from_float(cls: Type['FakeDynamicQuantHFConv1D'],
                   mod: torch.nn.Module,
                   qconfig: Optional[Dict[torch.nn.Module, Any]] = None,
                   activation_reduce_range: Optional[bool] = True,
                   **kwargs) -> Type['FakeDynamicQuantHFConv1D']:
        """Maps module from float to QAT-ready.

        Args:
            mod: Module to be mapped.
            qconfig: Quantization configuration.
            activation_reduce_range: Whether to reduce the range of activations.

        Returns:
            (Type['FakeDynamicQuantConv1d']): QAT-ready module.

        """
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + cls._FLOAT_MODULE.__name__

        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert mod.qconfig, 'Input float module must have a valid qconfig'
            qconfig = mod.qconfig

        qat_conv1d = cls(mod.nf,
                         mod.weight.shape[0],
                         activation_reduce_range=activation_reduce_range,
                         qconfig=qconfig,
                         **kwargs)

        qat_conv1d.weight = mod.weight
        qat_conv1d.bias = mod.bias

        return qat_conv1d

    def to_float(self) -> torch.nn.Module:
        """Maps module from QAT-ready to float.

        Returns:
            (torch.nn.Module): Float-based module.

        """

        weight = self.weight_fake_quant(self.weight)

        float_conv1d = transformers.modeling_utils.Conv1D(self.nf,
                                                          self.weight.shape[0])

        float_conv1d.weight = torch.nn.Parameter(weight)
        float_conv1d.bias = self.bias

        return float_conv1d


class FakeDynamicQuantHFConv1DForOnnx(FakeDynamicQuantHFConv1D):
    """Allows a QAT-ready huggingface's transformers Conv1D layer to be exported with ONNX.

    """

    def __init__(self, *args,  **kwargs):
        """Initializes a fake quantized Linear layer compatible with ONNX.
        
        """

        kwargs['activation_reduce_range'] = False
        kwargs['onnx_compatible'] = True

        super().__init__(*args, **kwargs)


def fake_dynamic_compute_logit(self,
                               hidden: torch.FloatTensor,
                               weight: torch.FloatTensor,
                               bias: torch.FloatTensor,
                               proj: torch.FloatTensor) -> torch.FloatTensor:
    """Translates `_compute_logit` from ProjectedAdaptiveLogSoftmax to a QAT-ready.

    Args:
        inputs: Input tensor.
        weight: Weight tensor.
        bias: Bias tensor.
        proj: Projection tensor.

    Returns:
        (torch.FloatTensor): Output logits.
    """

    if proj is None:
        fake_quant_hidden = self.hidden_fake_quant(hidden)
        fake_quant_weight = self.weight_fake_quant(weight)
        logit = F.linear(fake_quant_hidden, fake_quant_weight, bias=bias)
    else:
        logit = torch.einsum('bd,de,ev->bv', (hidden, proj, weight.t()))

        if bias is not None:
            logit = logit + bias

    return logit
