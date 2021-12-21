# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""ONNX-loading utilities that enable exports.
"""

import types
from os import environ
from typing import Tuple

from onnxruntime import (GraphOptimizationLevel, InferenceSession,
                         SessionOptions)

from archai.nlp.common.lazy_loader import load_from_checkpoint
from archai.nlp.compression.onnx.onnx_utils.forward import (
    crit_forward_memformer_onnx, forward_gpt2_onnx, forward_memformer_onnx)
from archai.nlp.models.model_base import ArchaiModel

# Constants available in onnxruntime
# that enables performance optimization
environ['OMP_NUM_THREADS'] = str(1)
environ['OMP_WAIT_POLICY'] = 'ACTIVE'


def load_from_onnx(onnx_model_path: str) -> InferenceSession:
    """Loads an ONNX-based model from file.

    Args:
        onnx_model_path: Path to the ONNX model file.

    Returns:
        (InferenceSession): ONNX inference session.

    """

    # Defines the ONNX loading options
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Creates an inference session
    session = InferenceSession(onnx_model_path, options)
    session.disable_fallback()

    return session


def load_from_torch_for_export(model_type: str,
                               torch_model_path: str) -> Tuple[ArchaiModel, dict]:
    """Loads a PyTorch-based model from checkpoint with export-ready.

    Args:
        model_type: Type of model to be loaded.
        torch_model_path: Path to the PyTorch model/checkpoint file.

    Returns:
        (ArchaiModel, dict): PyTorch model and its configuration.

    """

    # Loads the model
    model, model_config = load_from_checkpoint(model_type,
                                               torch_model_path,
                                               on_cpu=True,
                                               for_export=True)

    # Overrides forward functions if MemTransformerLM
    if model_type == 'mem_transformer':
        model.forward = types.MethodType(forward_memformer_onnx, model)
        model.crit.forward = types.MethodType(crit_forward_memformer_onnx, model.crit)

    # Overrides forward functions if HfGPT2
    if model_type == 'hf_gpt2':
        model = model.model
        model.forward = types.MethodType(forward_gpt2_onnx, model)

    if type(model_config['d_head']) is list:
        model_config['d_head'] = model_config['d_head'][0]
    if type(model_config['n_head']) is list:
        model_config['n_head'] = model_config['n_head'][0]

    # Puts to evaluation model to disable dropout
    model.eval()

    return model, model_config
