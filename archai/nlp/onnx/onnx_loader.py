# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""ONNX-loading utilities.
"""

from os import environ

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

from archai.nlp import logging_utils

logger = logging_utils.get_logger(__name__)


def load_from_onnx(onnx_model_path: str) -> InferenceSession:
    """Loads an ONNX-based model from file.

    Args:
        onnx_model_path: Path to the ONNX model file.

    Returns:
        (InferenceSession): ONNX inference session.

    """

    logger.info(f"Loading model: {onnx_model_path}")

    # Constants available in ONNXRuntime that enables performance optimization
    OMP_NUM_THREADS = 1
    environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
    environ["OMP_WAIT_POLICY"] = "ACTIVE"

    options = SessionOptions()
    options.intra_op_num_threads = OMP_NUM_THREADS
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    session = InferenceSession(onnx_model_path, sess_options=options)
    session.disable_fallback()

    return session
