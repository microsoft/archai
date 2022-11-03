# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Configuration objects needed by ONNX when performing
    export, quantization or any sort of operation.
"""

from collections import OrderedDict
from typing import Any, Mapping, Optional, Union

from transformers.file_utils import TensorType
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.gpt2.configuration_gpt2 import GPT2OnnxConfig
from transformers.onnx.config import OnnxConfig
from transformers.onnx.utils import compute_effective_axis_dimension

from archai.nlp.datasets.hf.tokenizer_utils.pre_trained_tokenizer import (
    ArchaiPreTrainedTokenizerFast,
)


class TransfoXLOnnxConfig(OnnxConfig):
    """Defines a TransformerXL-based configuration object for exporting to ONNX."""

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        """ONNX-based inputs structure.

        Returns:
            (Mapping[str, Mapping[int, str]]): ONNX-based inputs.

        """

        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("labels", {0: "batch", 1: "sequence"}),
            ]
        )

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        """ONNX-based outputs structure.

        Returns:
            (Mapping[str, Mapping[int, str]]): ONNX-based outputs.

        """

        return OrderedDict(
            [
                ("loss", {0: "batch", 1: "sequence"}),
                ("logits", {0: "batch", 1: "sequence"}),
            ]
        )

    def generate_dummy_inputs(
        self,
        tokenizer: Union[AutoTokenizer, ArchaiPreTrainedTokenizerFast],
        batch_size: Optional[int] = -1,
        seq_length: Optional[int] = -1,
        is_pair: Optional[bool] = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        """Generates dummy inputs for the ONNX exporter.

        Args:
            tokenizer: Pre-trained tokenizer.
            batch_size: Size of the batch (-1 for dynamic axis).
            seq_length: Size of the sequence (-1 for dynamic axis).
            is_pair: Whether the input is a pair (sentence 1, sentence 2).
            framework: Framework for the tensors that will be generated.

        Returns:
            (Mapping[str, Any]): Keyword arguments for the model's forward.

        """

        # When using dynamic axis, a fixed dimension is forward
        # to avoid ONNX optimizations
        batch_size = compute_effective_axis_dimension(
            batch_size,
            fixed_dimension=OnnxConfig.DEFAULT_FIXED_BATCH,
            num_token_to_add=0,
        )

        token_to_add = tokenizer.num_special_tokens_to_add(is_pair)
        seq_length = compute_effective_axis_dimension(
            seq_length,
            fixed_dimension=OnnxConfig.DEFAULT_FIXED_SEQUENCE,
            num_token_to_add=token_to_add,
        )

        dummy_input = [" ".join([tokenizer.unk_token]) * seq_length] * batch_size
        dummy_tokens = tokenizer(
            dummy_input,
            return_tensors=framework,
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        dummy_tokens["labels"] = dummy_tokens["input_ids"]

        return dict(dummy_tokens)
