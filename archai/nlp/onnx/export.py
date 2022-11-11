# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""ONNX-related export and validation.
"""

from itertools import chain
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from onnxruntime import InferenceSession, SessionOptions

from archai.nlp.onnx.config_utils.onnx_config_base import OnnxConfig
from archai.nlp.onnx.config_utils.gpt2_onnx_config import GPT2OnnxConfig, GPT2FlexOnnxConfig
from archai.nlp.onnx.export_utils import prepare_model_for_onnx, weight_sharing
from archai.nlp import logging_utils

logger = logging_utils.get_logger(__name__)

AVAILABLE_ONNX_CONFIGS = {
    "gpt2": GPT2OnnxConfig,
    "gpt2-flex": GPT2FlexOnnxConfig
}


def validate_onnx_outputs(
    onnx_config: OnnxConfig,
    reference_model: torch.nn.Module,
    onnx_model: Path,
    atol: float,
) -> None:
    """Validates the ONNX outputs.

    Args:
        onnx_config: ONNX configuration.
        reference_model: PyTorch reference model.
        onnx_model: Path to the ONNX model.
        atol: Tolerance value.

    """

    logger.info("Validating model ...")

    options = SessionOptions()
    session = InferenceSession(onnx_model.as_posix(), options)

    ref_inputs = onnx_config.generate_dummy_inputs()
    ref_outputs = reference_model(**ref_inputs)
    
    # Flattens the reference outputs
    ref_outputs_dict = {}
    for name, value in ref_outputs.items():
        if name == "past_key_values":
            name = "present"
        elif name == "logits":
            name = "probs"

        if isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                name_with_idx = f"{name}_{i}"
                ref_outputs_dict[name_with_idx] = v
        else:
            ref_outputs_dict[name] = value

    # Transforms the inputs into an ONNX compatible format
    onnx_inputs = {}
    for name, value in ref_inputs.items():
        if name == "past_key_values":
            name = "past"

        if isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                name_with_idx = f"{name}_{i}"
                onnx_inputs[name_with_idx] = v.numpy()
        else:
            onnx_inputs[name] = value.numpy()

    # Performs the ONNX inference session
    onnx_named_outputs = [output for output in onnx_config.outputs.keys()]
    onnx_outputs = session.run(onnx_named_outputs, onnx_inputs)

    # Checks whether subset of ONNX outputs is valid
    ref_outputs_set, onnx_outputs_set = set(ref_outputs_dict.keys()), set(onnx_config.outputs)
    if not onnx_outputs_set.issubset(ref_outputs_set):
        error = f"Unmatched outputs: {onnx_outputs_set} (ONNX) and {ref_outputs_set} (reference)"
        logger.error(error)
        raise ValueError(error)
    else:
        logger.debug(f"Matched outputs: {onnx_outputs_set}")

    # Checks whether shapes and values are within expected tolerance
    for name, ort_value in zip(onnx_config.outputs, onnx_outputs):
        logger.debug(f"Validating output: {name}")

        ref_value = ref_outputs_dict[name].detach().numpy()

        if not ort_value.shape == ref_value.shape:
            error = f"Unmatched shape: {ort_value.shape} (ONNX) and {ref_value.shape} (reference)"
            logger.error(error)
            raise ValueError(error)
        else:
            logger.debug(f"Matched shape: {ort_value.shape} (ONNX) and {ref_value.shape} (reference)")

        diff = np.amax(np.abs(ref_value - ort_value))
        if not np.allclose(ref_value, ort_value, atol=atol):
            error = f"Unmatched difference: {diff:.4e} > {atol}"
            logger.error(error)
            raise ValueError(error)
        else:
            logger.debug(f"Matched difference: {diff:.4e} < {atol}")


def export_to_onnx(
    model: torch.nn.Module,
    output_model_path: Path,
    task: Optional[str] = "causal-lm",
    use_past: Optional[bool] = True,
    batch_size: int = 2,
    seq_len: int = 8,
    share_weights: Optional[bool] = True,
    opset: Optional[int] = 11,
    atol: Optional[float] = 1e-4
) -> OnnxConfig:
    """Exports a pre-trained model to ONNX.

    Args:
        model: Instance of model to be exported.
        output_model_path: Path to save the exported model.
        task: Task identifier to use proper inputs/outputs.
        use_past: Whether past key/values (`use_cache`) should be used.
        batch_size: expected inference batch size
        seq_len: expected inference sequence length
        share_weights: Whether embedding/softmax weights should be shared.
        opset: Set of operations to use with ONNX.
        atol: Tolerance between input and exported model.

    Returns:
        (OnnxConfig): ONNX configuration of model that was exported.

    """

    logger.info(f"Exporting model: {output_model_path}")

    model_type = model.config.model_type
    available_configs = list(AVAILABLE_ONNX_CONFIGS.keys())
    assert model_type in available_configs, f"`model_type`: {model_type} is not supported for ONNX export."
    
    onnx_config = AVAILABLE_ONNX_CONFIGS[model_type](
        model.config, task=task, use_past=use_past,
        batch_size=batch_size, seq_len=seq_len
    )

    model = prepare_model_for_onnx(model, model_type)
    dynamic_axes = {name: axes for name, axes in chain(onnx_config.inputs.items(), onnx_config.outputs.items())}

    torch.onnx.export(model,
                      (onnx_config.generate_dummy_inputs(),),
                      f=output_model_path,
                      export_params=True,
                      input_names=list(onnx_config.inputs.keys()),
                      output_names=list(onnx_config.outputs.keys()),
                      dynamic_axes=dynamic_axes,
                      opset_version=opset,
                      do_constant_folding=True)
    validate_onnx_outputs(onnx_config, model, output_model_path, atol)

    if share_weights:
        weight_sharing(output_model_path, model_type)

    return onnx_config
