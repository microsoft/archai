# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""ONNX-related export and validation.
"""

import importlib
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from onnxruntime import InferenceSession, SessionOptions
from transformers.configuration_utils import PretrainedConfig
from transformers.file_utils import TensorType
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.onnx.config import OnnxConfig
from transformers.onnx.convert import export

from archai.nlp.datasets.hf.tokenizer_utils.pre_trained_tokenizer import (
    ArchaiPreTrainedTokenizerFast,
)
from archai.nlp.onnx.config_utils.gpt2_onnx_config import GPT2OnnxConfig
from archai.nlp.onnx.export_utils import prepare_model_for_onnx, weight_sharing
from archai.nlp import logging_utils

logger = logging_utils.get_logger(__name__)

AVAILABLE_ONNX_CONFIGS = {
    "gpt2": GPT2OnnxConfig,
    "gpt2-flex": GPT2OnnxConfig
}


def validate_onnx_outputs(
    config: OnnxConfig,
    tokenizer: Union[AutoTokenizer, ArchaiPreTrainedTokenizerFast],
    reference_model: torch.nn.Module,
    onnx_model: Path,
    onnx_named_outputs: List[str],
    atol: float,
) -> None:
    """Validates the ONNX outputs.

    Args:
        config: ONNX configuration.
        tokenizer: Pre-trained tokenizer.
        reference_model: PyTorch reference model.
        onnx_model: Path to the ONNX model.
        onnx_named_outputs: List of expected ONNX outputs.
        atol: Tolerance value.

    """

    logger.info("Validating ONNX model ...")

    options = SessionOptions()
    session = InferenceSession(onnx_model.as_posix(), options)

    ref_inputs = config.generate_dummy_inputs(tokenizer, framework=TensorType.PYTORCH)
    ref_outputs = reference_model(**ref_inputs)
    ref_outputs_dict = {}

    # print(type(ref_outputs))

    # Flattens the reference outputs
    for name, value in ref_outputs.items():
        # Overwriting the output name as 'present' since it is the name used for the ONNX ouputs
        # ('past_key_values' being taken for the ONNX inputs)
        if name == "past_key_values":
            name = "present"
        # Overwriting the output name as 'logits' since it is the proper prediction scores key
        elif name == "prediction_scores":
            name = "logits"
        # Overwriting the output name as 'loss' since it is the proper key (not 'losses')
        elif name == "losses":
            name = "loss"

        if isinstance(value, (list, tuple)):
            value = config.flatten_output_collection_property(name, value)
            ref_outputs_dict.update(value)
        else:
            ref_outputs_dict[name] = value

    # Transforms the inputs into an ONNX compatible format
    onnx_inputs = {}
    for name, value in ref_inputs.items():
        # if isinstance(value, (list, tuple)):
        #     value = config.flatten_output_collection_property(name, value)
        #     onnx_inputs.update(
        #         {tensor_name: pt_tensor.numpy() for tensor_name, pt_tensor in value.items()}
        #     )
        # else:
        onnx_inputs[name] = value.numpy()

    # Performs the ONNX inference session
    onnx_outputs = session.run(onnx_named_outputs, onnx_inputs)

    # Checks whether subset of ONNX outputs is valid
    ref_outputs_set, onnx_outputs_set = set(ref_outputs_dict.keys()), set(onnx_named_outputs)
    if not onnx_outputs_set.issubset(ref_outputs_set):
        logger.info(
            f"Incorrect outputs: {onnx_outputs_set} (ONNX) and {ref_outputs_set} (reference)"
        )
        raise ValueError(f"Unmatched outputs: {onnx_outputs_set.difference(ref_outputs_set)}")
    else:
        logger.info(f"Matched outputs: {onnx_outputs_set}")

    # Checks whether shapes and values are within expected tolerance
    for name, ort_value in zip(onnx_named_outputs, onnx_outputs):
        logger.info(f"Validating ONNX output: {name}")

        ref_value = ref_outputs_dict[name].detach().numpy()

        if not ort_value.shape == ref_value.shape:
            logger.info(
                f"Incorrect shape: {ort_value.shape} (ONNX) and {ref_value.shape} (reference)"
            )
            raise ValueError(
                f"Unmatched shape: {ort_value.shape} (ONNX) and {ref_value.shape} (reference)"
            )
        else:
            logger.info(f"Matched shape: {ort_value.shape} and {ref_value.shape}")

        if not np.allclose(ref_value, ort_value, atol=atol):
            logger.info(f"Incorrect tolerance: {atol}")
            raise ValueError(
                f"Unmatched value difference: {np.amax(np.abs(ref_value - ort_value))}"
            )
        else:
            logger.info(f"Matched tolerance: {atol}")


def export_to_onnx(
    model: torch.nn.Module,
    tokenizer: Union[AutoTokenizer, ArchaiPreTrainedTokenizerFast],
    output_model_path: Path,
    task: Optional[str] = "causal-lm",
    use_past: Optional[bool] = True,
    share_weights: Optional[bool] = True,
    opset: Optional[int] = 14,
    atol: Optional[float] = 1e-4,
) -> PretrainedConfig:
    """Exports a pre-trained model to ONNX.

    Args:
        model: Instance of model to be exported.
        tokenizer: Instance of tokenizer to generate dummy inputs.
        output_model_path: Path to save the exported model.
        task: Task identifier to use proper inputs/outputs.
        use_past: Whether past key/values (`use_cache`) should be used.
        share_weights: Whether embedding/softmax weights should be shared.
        opset: Set of operations to use with ONNX.
        atol: Tolerance between input and exported model.

    Returns:
        (PretrainedConfig): Configuration of model that was exported.

    """

    logger.info(f"Exporting to ONNX model: {output_model_path}")

    model.config.use_cache = use_past
    model.config.past_key_values = 2

    model_type = model.config.model_type.replace("-", "_")
    available_configs = list(AVAILABLE_ONNX_CONFIGS.keys())
    assert model_type in available_configs, f"`model_type` should be in {available_configs}."
    onnx_config = AVAILABLE_ONNX_CONFIGS[model_type](model.config, task=task, use_past=use_past)

    # config_module = importlib.import_module("archai.nlp.onnx.onnx_configs")
    # model_onnx_config = getattr(config_module, config_cls_name)
    # onnx_config = model_onnx_config

    model = prepare_model_for_onnx(model, model_type)
    _, onnx_outputs = export(tokenizer, model, onnx_config, opset, output_model_path)
    # validate_onnx_outputs(onnx_config, tokenizer, model, output_model_path, onnx_outputs, atol)

    if share_weights:
        weight_sharing(output_model_path, model_type)

    return model.config
