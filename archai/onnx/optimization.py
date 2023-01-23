# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

from onnx import load_model
from onnxruntime.transformers.onnx_model_gpt2 import Gpt2OnnxModel
from onnxruntime.transformers.optimizer import optimize_by_onnxruntime

from archai.common.file_utils import create_file_name_identifier
from archai.common.logger import Logger
from archai.onnx.config_utils.onnx_config_base import OnnxConfig
from archai.onnx.optimization_utils.fusion_options import FusionOptions

logger = Logger(source=__name__)

AVAILABLE_ONNX_MODELS = {"gpt2": Gpt2OnnxModel, "gpt2-flex": Gpt2OnnxModel}


def optimize_onnx(
    onnx_model_path: str,
    onnx_config: OnnxConfig,
    use_gpu: Optional[bool] = False,
    opt_level: Optional[int] = 1,
    only_ort: Optional[bool] = False,
    float16: Optional[bool] = False,
    input_int32: Optional[bool] = False,
) -> str:
    """Optimize an ONNX model using a combination of standard ORT-based optimization
    and additional transformer-based optimization.

    Args:
        onnx_model_path: Path to the ONNX model to be optimized.
        onnx_config: ONNX configuration of model to be optimized.
        use_gpu: Whether to use GPU during optimization.
        opt_level: Level of optimization.
        only_ort: Whether to only apply ORT optimization.
        float16: Whether to use graph with float16.
        input_int32: Whether to use inputs with int32.

    Returns:
        Path to the optimized ONNX model.

    """

    logger.info(f"Optimizing model: {onnx_model_path}")

    assert opt_level in [0, 1, 2, 99]
    ort_model_path = onnx_model_path

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
        ort_model_path = create_file_name_identifier(onnx_model_path, "-opt")
        optimize_by_onnxruntime(
            onnx_model_path,
            use_gpu=use_gpu,
            optimized_model_path=ort_model_path,
            opt_level=opt_level,
            disabled_optimizers=disabled_optimizers,
        )

    if not only_ort:
        model_type = onnx_config.config.model_type
        available_models = list(AVAILABLE_ONNX_MODELS.keys())
        assert model_type in available_models, f"`model_type`: {model_type} is not supported for `only_ort=False`."

        # Applies additional transformer-based optimization
        if onnx_config.is_ort_graph_optimizable:
            ort_model = load_model(ort_model_path)
            ort_model_path = create_file_name_identifier(onnx_model_path, "-opt")

            onnx_opt_model = AVAILABLE_ONNX_MODELS[model_type]
            options = FusionOptions(model_type)

            optimizer = onnx_opt_model(ort_model, *onnx_config.ort_graph_optimizer_args)
            optimizer.optimize(options)
            optimizer.topological_sort()

            if float16:
                ort_model_path = create_file_name_identifier(ort_model_path, "-fp16")
                optimizer.convert_float_to_float16(keep_io_types=True)

            if input_int32:
                optimizer.change_graph_inputs_to_int32()

            optimizer.save_model_to_file(ort_model_path)

    return ort_model_path
