# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import types
from os import environ
from pathlib import Path
from typing import Tuple

import torch
from onnxruntime import (GraphOptimizationLevel, InferenceSession,
                         SessionOptions)

from archai.nlp.nvidia_transformer_xl.mem_transformer import MemTransformerLM
from archai.nlp.nvidia_transformer_xl.onnx.onnx_utils.forward import (crit_forward_with_probs,
                                                                      forward_with_probs)

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


def load_from_pt(torch_model_path: str) -> Tuple[MemTransformerLM, dict]:
    """Loads a PyTorch-based model from checkpoint.

    Args:
        torch_model_path: Path to the PyTorch model/checkpoint file.

    Returns:
        (MemTransformerLM, dict): PyTorch model and its configuration.

    """

    # Loads the checkpoint
    # Note we are always enabling cache for usage of `past_key_values`
    checkpoint = torch.load(torch_model_path, map_location=torch.device('cpu'))
    checkpoint['model_config']['use_cache'] = True

    # Initializes the model

    # Added for compatibility with models trained with an 
    # older test branch which had this flag
    # TODO: Remove in the future
    if 'encoder_like' in checkpoint['model_config']:
        del checkpoint['model_config']['encoder_like']
    
    model = MemTransformerLM(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state'])

    # Overrides forward functions
    model.forward = types.MethodType(forward_with_probs, model)
    model.crit.forward = types.MethodType(crit_forward_with_probs, model.crit)

    # Puts to evaluation model to disable dropout
    model.eval()

    return model, checkpoint['model_config']
