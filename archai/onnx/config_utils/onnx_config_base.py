# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
from collections import OrderedDict
from typing import Any, Mapping, Optional, Tuple

import torch
from overrides import overrides
from overrides.enforce import EnforceOverrides
from transformers.configuration_utils import PretrainedConfig


class OnnxConfig(EnforceOverrides):
    """Base ONNX configuration.

    This class defines a base ONNX configuration for a specific task, which includes the
    input and output structure required for ONNX models, as well as additional properties
    and methods for handling ONNX Runtime graph optimization.

    """

    DEFAULT_TASK_OUTPUTS = {"causal-lm": OrderedDict({"probs": {0: "batch_size"}})}

    def __init__(
        self,
        config: PretrainedConfig,
        task: Optional[str] = "causal-lm",
    ) -> None:
        """Initialize the ONNX configuration by verifying whether the
            specified `task` is supported.

        Args:
            config: Configuration of the model being exported.
            task: Type of task that the exported model will be used for.

        """

        assert task in self.DEFAULT_TASK_OUTPUTS.keys(), f"`task`: {task} is not supported yet."

        self.config = config
        self.task = task

    @property
    def is_ort_graph_optimizable(self) -> bool:
        """Return whether configuration supports additional graph optimization."""

        return False

    @property
    def ort_graph_optimizer_args(self) -> Tuple[Any, ...]:
        """Return additional arguments used by the ORT graph optimizer."""

        return None

    def get_inputs(self) -> Mapping[str, Mapping[int, str]]:
        """Get the ONNX-based inputs structure.

        Returns:
            ONNX-based inputs.

        """

        return OrderedDict({"input_ids": {0: "batch_size", 1: "seq_len"}})

    def get_outputs(self) -> Mapping[str, Mapping[int, str]]:
        """Get the ONNX-based outputs structure.

        Returns:
            ONNX-based outputs.

        """

        return copy.deepcopy(self.DEFAULT_TASK_OUTPUTS[self.task])

    def generate_dummy_inputs(
        self, batch_size: Optional[int] = 2, seq_len: Optional[int] = 8
    ) -> Mapping[str, torch.Tensor]:
        """Generate dummy inputs for the ONNX exporter.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.

        Returns:
            Keyword arguments for the model's `forward()` function.

        """

        assert seq_len <= self.config.max_position_embeddings, (
            f"seq_len ({seq_len}) must be smaller than max_position_embeddings"
            f" ({self.config.max_position_embeddings})"
        )

        return {"input_ids": torch.zeros((batch_size, seq_len), dtype=torch.long)}


class OnnxConfigWithPast(OnnxConfig):
    """ONNX configuration with support for past key/values.

    This class is a subclass of `OnnxConfig` that adds the ability to use past key/values
    (also known as 'use_cache') in the model's ONNX export.

    """

    def __init__(
        self,
        config: PretrainedConfig,
        task: Optional[str] = "causal-lm",
        use_past: Optional[bool] = False,
        past_key_values: Optional[int] = 2,
    ) -> None:
        """Initialize the ONNX configuration with past key/values.

        Args:
            config: Model's configuration.
            task: Type of task that the exported model will be used for.
            use_past: Whether past key/values should be used.
            past_key_values: Number of past-related information (2 for key and values).

        """

        super().__init__(config, task=task)

        if use_past:
            self.config.use_cache = True
            self.config.past_key_values = past_key_values
        else:
            self.config.use_cache = False

        self.use_past = use_past

    @property
    def hidden_size(self) -> int:
        """Return the dimensionality of hidden units."""

        if not hasattr(self.config, "hidden_size"):
            raise AttributeError("Please override `hidden_size` with correct attribute.")

        return self.config.hidden_size

    @property
    def num_layers(self) -> int:
        """Return the number of layers."""

        if not hasattr(self.config, "num_layers"):
            raise AttributeError("Please override `num_layers` with correct attribute.")

        return self.config.num_layers

    @property
    def num_attention_heads(self) -> int:
        """Return the number of attention heads."""

        if not hasattr(self.config, "num_attention_heads"):
            raise AttributeError("Please override `num_attention_heads` with correct attribute.")

        return self.config.num_attention_heads

    @overrides
    def get_inputs(self) -> Mapping[str, Mapping[int, str]]:
        inputs = super().get_inputs()

        if self.use_past:
            for i in range(self.num_layers):
                # [past_key_values, batch_size, n_head, past_seq_len, d_head]
                inputs[f"past_{i}"] = {1: "batch_size", 3: "past_seq_len"}

        return inputs

    @overrides
    def get_outputs(self) -> Mapping[str, Mapping[int, str]]:
        outputs = super().get_outputs()

        if self.use_past:
            for i in range(self.num_layers):
                # [past_key_values, batch_size, n_head, total_seq_len, d_head]
                # Note that total_seq_len is seq_len + past_seq_len
                outputs[f"present_{i}"] = {1: "batch_size", 3: "total_seq_len"}

        return outputs

    @overrides
    def generate_dummy_inputs(
        self, batch_size: Optional[int] = 2, seq_len: Optional[int] = 8, past_seq_len: Optional[int] = 8
    ) -> Mapping[str, torch.Tensor]:
        """Generate dummy inputs for the ONNX exporter.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            past_seq_len: Past key/values sequence length.

        Returns:
            Keyword arguments for the model's `forward()` function.

        """

        assert seq_len + past_seq_len <= self.config.max_position_embeddings, (
            f"Dummy input generated size ({seq_len + past_seq_len}) must be smaller"
            f" than max_position_embeddings ({self.config.max_position_embeddings})."
        )

        dummy_inputs = super().generate_dummy_inputs(batch_size, seq_len)

        if self.use_past:
            # [past_key_values, batch_size, n_head, past_seq_len, d_head]
            dummy_inputs["past_key_values"] = tuple(
                [
                    torch.zeros(
                        self.config.past_key_values,
                        batch_size,
                        self.num_attention_heads,
                        past_seq_len,
                        self.hidden_size // self.num_attention_heads,
                    )
                    for _ in range(self.num_layers)
                ]
            )

        return dummy_inputs
