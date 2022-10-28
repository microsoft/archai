# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Pipeline for performing PyTorch-based Quantization Aware Training (QAT).
"""

from typing import Any, Dict, Optional

import torch
import transformers

from archai.nlp.compression.quantization.modules import (
    FakeDynamicQuantConv1d,
    FakeDynamicQuantConv1dForOnnx,
    FakeDynamicQuantHFConv1D,
    FakeDynamicQuantHFConv1DForOnnx,
    FakeDynamicQuantLinear,
    FakeDynamicQuantLinearForOnnx,
    FakeQuantEmbedding,
    FakeQuantEmbeddingForOnnx,
)

DYNAMIC_QAT_MODULE_MAP = {
    torch.nn.Embedding: FakeQuantEmbedding,
    torch.nn.Linear: FakeDynamicQuantLinear,
    torch.nn.Conv1d: FakeDynamicQuantConv1d,
    transformers.modeling_utils.Conv1D: FakeDynamicQuantHFConv1D,
}
ONNX_DYNAMIC_QAT_MODULE_MAP = {
    torch.nn.Embedding: FakeQuantEmbeddingForOnnx,
    torch.nn.Linear: FakeDynamicQuantLinearForOnnx,
    torch.nn.Conv1d: FakeDynamicQuantConv1dForOnnx,
    transformers.modeling_utils.Conv1D: FakeDynamicQuantHFConv1DForOnnx,
}


def qat_to_float_modules(model: torch.nn.Module) -> torch.nn.Module:
    """Changes QAT-ready modules to float-based modules.

    Args:
        model: QAT-ready module.

    Returns:
        (torch.nn.Module): Float-based module.

    """

    for name in list(model._modules):
        module = model._modules[name]

        if hasattr(module, "to_float"):
            model._modules[name] = module.to_float()
        else:
            qat_to_float_modules(module)

    return model


def float_to_qat_modules(
    model: torch.nn.Module,
    module_mapping: Optional[Dict[torch.nn.Module, torch.nn.Module]] = DYNAMIC_QAT_MODULE_MAP,
    qconfig: Optional[Dict[torch.nn.Module, Any]] = None,
    **kwargs
) -> torch.nn.Module:
    """Changes float-based modules to QAT-ready modules.

    Args:
        model: Float-based module.
        module_mapping: Maps between float and QAT-ready modules.
        qconfig: Quantization configuration.

    Returns:
        (torch.nn.Module): QAT-ready module.

    """

    for name in list(model._modules):
        module = model._modules[name]

        if type(module) in module_mapping:
            if not hasattr(module, "qconfig"):
                module.qconfig = qconfig

            model._modules[name] = module_mapping[type(module)].from_float(
                module, qconfig, **kwargs
            )

        else:
            float_to_qat_modules(
                module, module_mapping=module_mapping, qconfig=qconfig, **kwargs
            )

    return model


def prepare_with_qat(
    model: torch.nn.Module,
    onnx_compatible: Optional[bool] = False,
    backend: Optional[str] = "qnnpack",
    **kwargs
) -> torch.nn.Module:
    """Prepares a float-based model and inserts QAT-based modules and configurations.

    Args:
        model: Float-based module.
        onnx_compatible: Whether QAT-ready module is compatible with ONNX.
        backend: Quantization backend.

    Returns:
        (torch.nn.Module): QAT-ready module.

    """

    qconfig = torch.quantization.get_default_qat_qconfig(backend)
    module_mapping = ONNX_DYNAMIC_QAT_MODULE_MAP if onnx_compatible else DYNAMIC_QAT_MODULE_MAP

    float_to_qat_modules(model, module_mapping=module_mapping, qconfig=qconfig, **kwargs)

    return model
