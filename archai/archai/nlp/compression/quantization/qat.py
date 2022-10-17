# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Pipeline for performing PyTorch-based Quantization Aware Training (QAT).
"""

from typing import Any, Dict, Optional

import torch
import transformers

from archai.nlp.compression.quantization.modules import (
    FakeDynamicQuantConv1d, FakeDynamicQuantConv1dForOnnx,
    FakeDynamicQuantHFConv1D, FakeDynamicQuantHFConv1DForOnnx,
    FakeDynamicQuantLinear, FakeDynamicQuantLinearForOnnx, FakeQuantEmbedding,
    FakeQuantEmbeddingForOnnx, fake_dynamic_compute_logit)
from archai.nlp.compression.quantization.quantizers import FakeDynamicQuant
from archai.nlp.models.mem_transformer.mem_transformer_utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax

# Maps between standard and ONNX modules
DYNAMIC_QAT_MODULE_MAPPING = {
    torch.nn.Embedding: FakeQuantEmbedding,
    torch.nn.Linear: FakeDynamicQuantLinear,
    torch.nn.Conv1d: FakeDynamicQuantConv1d,
    transformers.modeling_utils.Conv1D: FakeDynamicQuantHFConv1D
    
}
DYNAMIC_QAT_MODULE_MAPPING_FOR_ONNX = {
    torch.nn.Embedding: FakeQuantEmbeddingForOnnx,
    torch.nn.Linear: FakeDynamicQuantLinearForOnnx,
    torch.nn.Conv1d: FakeDynamicQuantConv1dForOnnx,
    transformers.modeling_utils.Conv1D: FakeDynamicQuantHFConv1DForOnnx
}

# Adds placeholder for changing `_compute_logit`
COMPUTE_LOGIT = ProjectedAdaptiveLogSoftmax._compute_logit


def qat_to_float_modules(model: torch.nn.Module) -> torch.nn.Module:
    """Changes QAT-ready modules to float-based modules.

    Args:
        model: QAT-ready module.
    
    Returns:
        (torch.nn.Module): Float-based module.

    """

    for name in list(model._modules):
        module = model._modules[name]            

        # Checks whether module can be mapped back to float
        if hasattr(module, 'to_float'):
            # Maps the module back to float
            model._modules[name] = module.to_float()
        else:
            # If module can not be mapped, recursively calls the function
            qat_to_float_modules(module)

    ProjectedAdaptiveLogSoftmax._compute_logit = COMPUTE_LOGIT

    return model


def float_to_qat_modules(model: torch.nn.Module,
                         module_mapping: Optional[Dict[torch.nn.Module, torch.nn.Module]] = DYNAMIC_QAT_MODULE_MAPPING,
                         qconfig: Optional[Dict[torch.nn.Module, Any]] = None,
                         **kwargs) -> torch.nn.Module:
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
            # Checks whether `qconfig` has already been inserted or not
            if not hasattr(module, 'qconfig'):
                module.qconfig = qconfig

            # Maps from float to QAT
            model._modules[name] = module_mapping[type(module)].from_float(module, qconfig, **kwargs)

        else:
            # If there is no module to be mapped, recursively calls the function
            float_to_qat_modules(module,
                                 module_mapping=module_mapping,
                                 qconfig=qconfig,
                                 **kwargs)

    # Applies fake quantization to `ProjectedAdaptiveLogSoftmax` as well
    ProjectedAdaptiveLogSoftmax.hidden_fake_quant = FakeDynamicQuant(reduce_range=False,
                                                                     onnx_compatible=True)
    ProjectedAdaptiveLogSoftmax.weight_fake_quant = FakeDynamicQuant(dtype=torch.qint8,
                                                                     reduce_range=False,
                                                                     onnx_compatible=True)
    ProjectedAdaptiveLogSoftmax._compute_logit = fake_dynamic_compute_logit

    return model


def prepare_with_qat(model: torch.nn.Module,
                     onnx_compatible: Optional[bool] = False,
                     backend: Optional[str] = 'qnnpack',
                     **kwargs) -> torch.nn.Module:
    """Prepares a float-based model and inserts QAT-based modules and configurations.

    Args:
        model: Float-based module.
        onnx_compatible: Whether QAT-ready module is compatible with ONNX.
        backend: Quantization backend.

    Returns:
        (torch.nn.Module): QAT-ready module.

    """

    # Gathers the `qconfig` and appropriate modules mappings
    qconfig = torch.quantization.get_default_qat_qconfig(backend)
    mappings = DYNAMIC_QAT_MODULE_MAPPING_FOR_ONNX if onnx_compatible else DYNAMIC_QAT_MODULE_MAPPING

    # Ensures that the model is QAT-ready
    float_to_qat_modules(model,
                         module_mapping=mappings,
                         qconfig=qconfig,
                         **kwargs)

    return model
