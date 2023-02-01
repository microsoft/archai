# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from os import environ
from typing import List, Optional

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

from archai.common.ordered_dict_logger import OrderedDictLogger

logger = OrderedDictLogger(source=__name__)


def load_from_onnx(onnx_model_path: str, providers: Optional[List[str]] = None) -> InferenceSession:
    """Load an ONNX-based model from file.

    This function loads an ONNX-based model from the specified file path and
    returns an ONNX inference session. Performance optimization constants are set as well.

    Args:
        onnx_model_path: Path to the ONNX model file.
        providers: List of providers to use for inference.

    Returns:
        ONNX inference session.

    """

    logger.info(f"Loading model: {onnx_model_path}")

    # Constants available in ONNXRuntime that enables performance optimization
    OMP_NUM_THREADS = 1
    environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
    environ["OMP_WAIT_POLICY"] = "ACTIVE"

    options = SessionOptions()
    options.intra_op_num_threads = OMP_NUM_THREADS
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = providers or ["CPUExecutionProvider"]

    session = InferenceSession(onnx_model_path, sess_options=options, providers=providers)
    session.disable_fallback()

    return session
