# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""ONNX optimization-related tools.
"""

from pathlib import Path
from typing import Optional, Sized

from onnx import load_model
from onnxruntime.transformers.optimizer import optimize_by_onnxruntime
from transformers.configuration_utils import PretrainedConfig

from onnxruntime.transformers.onnx_model_gpt2 import Gpt2OnnxModel

from archai.nlp.onnx.optimization_utils.fusion_options import FusionOptions
from archai.nlp import logging_utils
from archai.nlp.file_utils import create_file_name_identifier

logger = logging_utils.get_logger(__name__)

AVAILABLE_ONNX_MODELS = {"gpt2": Gpt2OnnxModel, "gpt2-flex": Gpt2OnnxModel}


def optimize_onnx(
    onnx_model_path: str,
    model_config: PretrainedConfig,
    use_gpu: Optional[bool] = False,
    opt_level: Optional[int] = 1,
    only_ort: Optional[bool] = False,
    float16: Optional[bool] = False,
    input_int32: Optional[bool] = False,
) -> Path:
    """Optimizes an ONNX model.

    Args:
        onnx_model_path: Path to the ONNX model to be optimized.
        model_config: Configuration of model to be optimized.
        use_gpu: Whether to use GPU during optimization.
        opt_level: Level of optimization.
        only_ort: Whether to only apply ORT optimization.
        float16: Whether to use graph with float16.
        input_int32: Whether to use inputs with int32.

    Returns:
        (Path): Path to the optimized ONNX model.

    """

    logger.info(f"Optimizing ONNX model: {onnx_model_path}")

    assert opt_level in [0, 1, 2, 99]
    ort_model_path = None

    # Applies standard ORT-based optimization
    if opt_level > 0:
        disabled_optimizers = []

        if opt_level > 1:
            # Disables some optimizers that might influence shape inference/attention fusion
            if not only_ort:
                disabled_optimizers = [
                    "MatMulScaleFusion",
                    "MatMulAddFusion",
                    "SimplifiedLayerNormFusion",
                    "GemmActivationFusion",
                    "BiasSoftmaxFusion",
                ]

        # Performs the standard ORT optimization
        ort_model_path = create_file_name_identifier(Path(onnx_model_path), "_ort")
        optimize_by_onnxruntime(
            onnx_model_path.as_posix(),
            use_gpu=use_gpu,
            optimized_model_path=str(ort_model_path),
            opt_level=opt_level,
            disabled_optimizers=disabled_optimizers,
        )

    # Applies additional transformer-based optimization
    if not only_ort:
        ort_model = load_model(ort_model_path or onnx_model_path)
        ort_model_path = create_file_name_identifier(Path(onnx_model_path), "_opt")

        model_type = model_config.model_type
        available_models = list(AVAILABLE_ONNX_MODELS.keys())
        assert model_type in available_models, f"`model_type` should be in {available_models}."
        onnx_opt_model = AVAILABLE_ONNX_MODELS[model_type]

        # Ensures that tuple of arguments is correct for the optimizer
        optimizer_args = (ort_model,)

        if model_type in ["gpt2", "gpt2-flex"]:
            n_heads, h_size = model_config.num_attention_heads, model_config.hidden_size

            # Quick fix to unlist elements (TODO: remove this altogether from config)
            n_heads = n_heads[0] if isinstance(n_heads, Sized) else n_heads
            h_size = h_size[0] if isinstance(h_size, Sized) else h_size

            optimizer_args += (n_heads, h_size)
        
        optimizer = onnx_opt_model(*optimizer_args)
        options = FusionOptions(model_type)

        optimizer.optimize(options)
        optimizer.topological_sort()

        if float16:
            ort_model_path = create_file_name_identifier(Path(onnx_model_path), "_opt_fp16")
            optimizer.convert_float_to_float16(keep_io_types=True)

        if input_int32:
            optimizer.change_graph_inputs_to_int32()

    optimizer.save_model_to_file(ort_model_path.as_posix())

    return ort_model_path
