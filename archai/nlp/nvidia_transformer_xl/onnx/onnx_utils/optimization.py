# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from typing import Optional

from onnx import load_model
from onnxruntime.transformers.optimizer import optimize_by_onnxruntime

from archai.nlp.nvidia_transformer_xl.onnx.onnx_utils.load import create_file_name_identifier
from archai.nlp.nvidia_transformer_xl.onnx.onnx_utils.opt.fusion_options import FusionOptions
from archai.nlp.nvidia_transformer_xl.onnx.onnx_utils.opt.model import MemTransformerLMOnnxModel


def optimize_onnx(onnx_model_path: str,
                  use_gpu: Optional[bool] = False,
                  opt_level: Optional[int] = 0,
                  only_ort: Optional[bool] = False,
                  float16: Optional[bool] = False,
                  input_int32: Optional[bool] = False) -> Path:
    """Optimizes an ONNX model.

    Args:
        onnx_model_path: Path to the ONNX model to be optimized.
        use_gpu: Whether to use GPU during optimization.
        opt_level: Level of optimization.
        only_ort: Whether to only apply ORT optimization.
        float16: Whether to use graph with float16.
        input_int32: Whether to use inputs with int32.

    Returns:
        (Path): Path to the optimized ONNX model.

    """

    assert opt_level in [0, 1, 2, 99]
    ort_model_path = None

    # Applies standard ORT-based optimization
    if opt_level > 0:
        disabled_optimizers = []

        if opt_level > 1:
            # Disables some optimizers that might influence shape inference/attention fusion.
            if not only_ort:
                disabled_optimizers = ['MatMulScaleFusion', 'MatMulAddFusion'
                                       'SimplifiedLayerNormFusion', 'GemmActivationFusion',
                                       'BiasSoftmaxFusion']

        # Performs the standard ORT optimization
        ort_model_path = create_file_name_identifier(Path(onnx_model_path), '_ort')
        optimize_by_onnxruntime(onnx_model_path,
                                use_gpu=use_gpu,
                                optimized_model_path=str(ort_model_path),
                                opt_level=opt_level,
                                disabled_optimizers=disabled_optimizers)

    # Applies additional transformer-based optimization
    if not only_ort:
        # Loads the ORT-optimized model, optimizer and fusion options
        ort_model = load_model(ort_model_path or onnx_model_path)
        ort_model_path = create_file_name_identifier(Path(onnx_model_path), '_opt')
        optimizer = MemTransformerLMOnnxModel(ort_model)
        options = FusionOptions()

        # Optimizes the model
        optimizer.optimize(options)

        # Applies float16 to the model
        if float16:
            ort_model_path = create_file_name_identifier(Path(onnx_model_path), '_opt_fp16')
            optimizer.convert_float_to_float16(keep_io_types=True)

        # Applies int32 to the model inputs
        if input_int32:
            optimizer.change_graph_inputs_to_int32()

    # Saves the model to file
    optimizer.save_model_to_file(str(ort_model_path))

    return ort_model_path
