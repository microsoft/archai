# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""ONNX-related optimization helpers and utilities.
"""

from pathlib import Path
from typing import Optional

from onnx import load_model
from onnxruntime.transformers.optimizer import optimize_by_onnxruntime

from archai.common.utils import create_file_name_identifier
from archai.nlp.models.model_loader import load_onnx_model
from archai.nlp.compression.onnx.onnx_utils.fusion_options import FusionOptions


def optimize_onnx(model_type: str,
                  onnx_model_path: str,
                  num_heads: Optional[int] = 8,
                  use_gpu: Optional[bool] = False,
                  opt_level: Optional[int] = 0,
                  only_ort: Optional[bool] = False,
                  float16: Optional[bool] = False,
                  input_int32: Optional[bool] = False) -> Path:
    """Optimizes an ONNX model.

    Args:
        model_type: Type of model to be optimized.
        onnx_model_path: Path to the ONNX model to be optimized.
        num_heads: Number of attention heads.
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

        # Puts the arguments for the optimizer
        optimizer_args = (ort_model, )
        if model_type in ['hf_gpt2', 'hf_gpt2_flex']:
            # Adds `hidden_size` as zero just for retro-compatibility
            optimizer_args += (num_heads, 0)
            
        optimizer = load_onnx_model(model_type, *optimizer_args)
        options = FusionOptions(model_type)

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
