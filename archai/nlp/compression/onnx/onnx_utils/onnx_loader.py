# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""ONNX-loading utilities that enable exports.
"""

import copy
import types
from os import environ
from typing import Any, Dict, Sized, Tuple

from onnxruntime import (GraphOptimizationLevel, InferenceSession,
                         SessionOptions)
from onnxruntime.transformers import quantize_helper

from archai.nlp.models.model_loader import load_model_from_checkpoint, load_model_from_config
from archai.nlp.compression.onnx.onnx_utils.forward import (crit_forward_mem_transformer_onnx,
                                                            forward_hf_gpt2_onnx,
                                                            forward_mem_transformer_onnx)
from archai.nlp.models.model_base import ArchaiModel

# ONNX-loading constants
OMP_NUM_THREADS = 1
OMP_WAIT_POLICY = 'ACTIVE'

# Constants available in onnxruntime
# that enables performance optimization
environ['OMP_NUM_THREADS'] = str(OMP_NUM_THREADS)
environ['OMP_WAIT_POLICY'] = OMP_WAIT_POLICY


def load_from_onnx(onnx_model_path: str) -> InferenceSession:
    """Loads an ONNX-based model from file.

    Args:
        onnx_model_path: Path to the ONNX model file.

    Returns:
        (InferenceSession): ONNX inference session.

    """

    # Defines the ONNX loading options
    options = SessionOptions()
    options.intra_op_num_threads = OMP_NUM_THREADS
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Creates an inference session
    session = InferenceSession(onnx_model_path, options)
    session.disable_fallback()

    return session


def _prepare_export(model: ArchaiModel,
                    model_config: Dict[str, Any],
                    model_type: str) -> ArchaiModel:
    """Prepares a PyTorch model with export-ready.

    Args:
        model: PyTorch model.
        model_config: Model configuration.
        model_type: Type of model.

    Returns:
        (ArchaiModel): Export-ready PyTorch model.

    """

    # Overrides forward functions if MemTransformerLM
    if model_type == 'mem_transformer':
        model.forward = types.MethodType(forward_mem_transformer_onnx, model)
        model.crit.forward = types.MethodType(crit_forward_mem_transformer_onnx, model.crit)

    # Overrides forward functions if HfGPT2
    if model_type in ['hf_gpt2', 'hf_gpt2_flex']:
        model = model.model
        model.forward = types.MethodType(forward_hf_gpt2_onnx, model)

        for layer in model.transformer.h:
            quantize_helper.conv1d_to_linear(layer.mlp)

    if isinstance(model_config['d_head'], Sized):
        model_config['d_head'] = model_config['d_head'][0]
    if isinstance(model_config['n_head'], Sized):
        model_config['n_head'] = model_config['n_head'][0]
    if model_config['d_head'] < 0:
        model_config['d_head'] = model_config['d_model'] // model_config['n_head']

    # Puts to evaluation model to disable dropout
    model.eval()

    return model, model_config


def load_from_config_for_export(model_type: str,
                                model_config: Dict[str, Any]) -> Tuple[ArchaiModel, Dict[str, Any]]:
    """Loads a PyTorch-based model from configuration with export-ready.

    Args:
        model_type: Type of model to be loaded.
        model_config: Model configuration.

    Returns:
        (ArchaiModel, Dict[str, Any]): Export-ready PyTorch model and its configuration.

    """

    # Copies model's configuration to prevent changing the original one
    export_model_config = copy.deepcopy(model_config)
    export_model_config['use_cache'] = True

    # Loads the model from configuration
    model = load_model_from_config(model_type, export_model_config)

    # Prepares the model for export
    model, export_model_config = _prepare_export(model, export_model_config, model_type)

    return model, export_model_config


def load_from_torch_for_export(model_type: str,
                               torch_model_path: str) -> Tuple[ArchaiModel, Dict[str, Any]]:
    """Loads a PyTorch-based model from checkpoint with export-ready.

    Args:
        model_type: Type of model to be loaded.
        torch_model_path: Path to the PyTorch model/checkpoint file.

    Returns:
        (ArchaiModel, Dict[str, Any]): Export-ready PyTorch model and its configuration.

    """

    # Loads the model
    model, model_config, _ = load_model_from_checkpoint(model_type,
                                                        torch_model_path,
                                                        on_cpu=True,
                                                        for_export=True)

    # Prepares the model for export
    model, model_config = _prepare_export(model, model_config, model_type)

    return model, model_config
