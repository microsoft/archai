# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import types
from os import environ
from pathlib import Path
from typing import Tuple

from onnxruntime import (GraphOptimizationLevel, InferenceSession,
                         SessionOptions)

from archai.nlp.nvidia_transformer_xl.models.archai_model import ArchaiModel
from archai.nlp.nvidia_transformer_xl.models.available_models import AVAILABLE_MODELS
from archai.nlp.nvidia_transformer_xl.onnx.onnx_utils.forward import (crit_forward_memformer_onnx, forward_gpt2_onnx,
                                                                      forward_memformer_onnx)

# Constants available in onnxruntime
# that enables performance optimization
environ['OMP_NUM_THREADS'] = str(1)
environ['OMP_WAIT_POLICY'] = 'ACTIVE'


def create_file_name_identifier(file_name: Path,
                                identifier: str) -> Path:
    """Adds an identifier (suffix) to the end of the file name.

    Args:
        file_name: Path to have a suffix added.
        identifier: Identifier to be added to file_name.

    Returns:
        (Path): Path with `file_name` plus added identifier.

    """

    return file_name.parent.joinpath(file_name.stem + identifier).with_suffix(file_name.suffix)


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


def load_from_pt(model_type: str, torch_model_path: str) -> Tuple[ArchaiModel, dict]:
    """Loads a PyTorch-based model from checkpoint.

    Args:
        model_type: Type of model to be loaded.
        torch_model_path: Path to the PyTorch model/checkpoint file.

    Returns:
        (ArchaiModel, dict): PyTorch model and its configuration.

    """

    # Loads the model
    model, model_config, _ = ArchaiModel.load_model(AVAILABLE_MODELS[model_type],
                                                    torch_model_path,
                                                    on_cpu=False,
                                                    for_export=True)

    # Overrides forward functions if MemTransformerLM
    if model_type == 'mem_transformer':
        model.forward = types.MethodType(forward_memformer_onnx, model)
        model.crit.forward = types.MethodType(crit_forward_memformer_onnx, model.crit)

    if model_type == 'hf_gpt2':
        model.forward = types.MethodType(forward_gpt2_onnx, model)

    # Puts to evaluation model to disable dropout
    model.eval()

    return model, model_config
