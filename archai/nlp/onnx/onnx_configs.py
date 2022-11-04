# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Configuration objects needed by ONNX when performing
    export, quantization or any sort of operation.
"""

import torch

from collections import OrderedDict
from typing import Any, Mapping, Optional, Union

from transformers.file_utils import TensorType
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.onnx.config import OnnxConfig, OnnxConfigWithPast
from transformers.onnx.utils import compute_effective_axis_dimension

from archai.nlp.datasets.hf.tokenizer_utils.pre_trained_tokenizer import (
    ArchaiPreTrainedTokenizerFast,
)


class GPT2OnnxConfig(OnnxConfigWithPast):
    """Defines a GPT2-based configuration object for exporting to ONNX."""

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        input_ids = [('input_ids', {0: 'batch_size', 1: 'seq_len'})]

        # Shape of past states
        # [past_key_values, batch_size, n_head, past_seq_len, d_head]
        past_key_values = [(f'past_{i}', {1: 'batch_size', 3: 'past_seq_len'}) for i in range(self._config.n_layer)]

        return OrderedDict(input_ids + past_key_values)

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        probs = [('probs', {0: 'batch_size'})]

        # Shape of present states (past states when outputting)
        # [past_key_values, batch_size, n_head, total_seq_len, d_head]
        # Note that total_seq_len is seq_len + past_seq_len
        present_key_values = [(f'present_{i}', {1: 'batch_size', 3: 'total_seq_len'}) for i in range(self._config.n_layer)]

        return OrderedDict(probs + present_key_values)

    # @property
    # def inputs(self) -> Mapping[str, Mapping[int, str]]:
    #     """ONNX-based inputs structure.

    #     Returns:
    #         (Mapping[str, Mapping[int, str]]): ONNX-based inputs.

    #     """

    #     common_inputs = OrderedDict({"input_ids": {0: "batch", 1: "sequence"}})
    #     if self.use_past:
    #         self.fill_with_past_key_values_(common_inputs, direction="inputs")

    #     return common_inputs

    @property
    def num_layers(self) -> int:
        return self._config.n_layer

    @property
    def num_attention_heads(self) -> int:
        return self._config.n_head

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

        common_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        ordered_inputs = OrderedDict({"input_ids": common_inputs["input_ids"]})

        if self.use_past:
            batch, seqlen = common_inputs["input_ids"].shape
            # Not using the same length for past_key_values
            past_key_values_length = seqlen + 2
            past_shape = (
                self._config.past_key_values,
                batch,
                self.num_attention_heads,
                past_key_values_length,
                self._config.hidden_size // self.num_attention_heads,
            )
            ordered_inputs["past_key_values"] = tuple([torch.zeros(past_shape) for _ in range(self.num_layers)])

        return ordered_inputs


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
