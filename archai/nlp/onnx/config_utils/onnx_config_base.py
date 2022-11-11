# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""ONNX-based configuration.
"""

import copy
from collections import OrderedDict
from typing import Any, Mapping, Optional, Tuple

import torch
from transformers.configuration_utils import PretrainedConfig


class OnnxConfig:
    """Implements the base ONNX configuration."""
    DEFAULT_TASK_OUTPUTS = {"causal-lm": OrderedDict({"probs": {0: "batch_size"}})}

    def __init__(
        self,
        config: PretrainedConfig,
        task: Optional[str] = "causal-lm",
        batch_size: int = 2,
        seq_len: int = 8,
    ) -> None:
        """Initializes by verifying whether `task` is supported.

        Args:
            config: Model's configuration.
            task: Type of task that the exported model will be used.

        """

        assert task in self.DEFAULT_TASK_OUTPUTS.keys(), f"`task`: {task} is not supported yet."

        self.config = config
        self.task = task
        self.batch_size = batch_size
        self.seq_len = seq_len

    @property
    def is_ort_graph_optimizable(self) -> bool:
        """Supports additional ONNX Runtime graph optimization.

        Returns:
            (bool): Whether configuration supports additional graph optimization.

        """

        return False

    @property
    def ort_graph_optimizer_args(self) -> Tuple[Any, ...]:
        """ONNX Runtime additional graph optimization arguments.

        Returns:
            (Tuple[Any, ...]): Additional arguments used by the ORT graph optimizer.

        """

        return None

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        """ONNX-based inputs structure.

        Returns:
            (Mapping[str, Mapping[int, str]]): ONNX-based inputs.

        """

        return OrderedDict({"input_ids": {0: "batch_size", 1: "seq_len"}})

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        """ONNX-based outputs structure.

        Returns:
            (Mapping[str, Mapping[int, str]]): ONNX-based outputs.

        """

        return copy.deepcopy(self.DEFAULT_TASK_OUTPUTS[self.task])

    def generate_dummy_inputs(self) -> Mapping[str, torch.Tensor]:
        """Generates dummy inputs for the ONNX exporter.

        Returns:
            (Mapping[str, Any]): Keyword arguments for the model's forward.

        """

        return {"input_ids": torch.zeros((self.batch_size, self.seq_len), dtype=torch.long)}


class OnnxConfigWithPast(OnnxConfig):
    """Implements the base ONNX configuration with support for past key/values."""

    def __init__(
        self,
        config: PretrainedConfig,
        task: Optional[str] = "causal-lm",
        use_past: Optional[bool] = False,
        past_key_values: Optional[int] = 2,
        batch_size: int = 2,
        seq_len: int = 8
    ) -> None:
        """Overrides initialization and defines whether past key/values are used.

        Args:
            config: Model's configuration.
            task: Type of task that the exported model will be used.
            use_past: Whether past key/values (`use_cache`) should be used.
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
        """Dimensionality of hidden units.

        Returns:
            (int): Hidden units size.

        """

        if not hasattr(self.config, "hidden_size"):
            raise AttributeError("Please override `hidden_size` with correct attribute.")

        return self.config.hidden_size

    @property
    def num_layers(self) -> int:
        """Number of layers.

        Returns:
            (int): Number of layers.

        """

        if not hasattr(self.config, "num_layers"):
            raise AttributeError("Please override `num_layers` with correct attribute.")

        return self.config.num_layers

    @property
    def num_attention_heads(self) -> int:
        """Number of attention heads.

        Returns:
            (int): Number of attention heads.

        """

        if not hasattr(self.config, "num_attention_heads"):
            raise AttributeError("Please override `num_attention_heads` with correct attribute.")

        return self.config.num_attention_heads

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        """ONNX-based inputs structure.

        Returns:
            (Mapping[str, Mapping[int, str]]): ONNX-based inputs.

        """

        inputs = super().inputs

        if self.use_past:
            for i in range(self.num_layers):
                # [past_key_values, batch_size, n_head, past_seq_len, d_head]
                inputs[f"past_{i}"] = {1: "batch_size", 3: "past_seq_len"}

        return inputs

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        """ONNX-based outputs structure.

        Returns:
            (Mapping[str, Mapping[int, str]]): ONNX-based outputs.

        """

        outputs = super().outputs

        if self.use_past:
            for i in range(self.num_layers):
                # [past_key_values, batch_size, n_head, total_seq_len, d_head]
                # Note that total_seq_len is seq_len + past_seq_len
                outputs[f"present_{i}"] = {1: "batch_size", 3: "total_seq_len"}

        return outputs

    def generate_dummy_inputs(self) -> Mapping[str, torch.Tensor]:
        """Generates dummy inputs for the ONNX exporter.

        Returns:
            (Mapping[str, Any]): Keyword arguments for the model's forward.

        """

        dummy_inputs = super().generate_dummy_inputs()

        if self.use_past:
            # [past_key_values, batch_size, n_head, past_seq_len, d_head]
            dummy_inputs["past_key_values"] = tuple(
                [
                    torch.zeros(
                        self.config.past_key_values,
                        self.batch_size,
                        self.num_attention_heads,
                        self.seq_len,
                        self.hidden_size // self.num_attention_heads,
                    )
                    for _ in range(self.num_layers)
                ]
            )

        return dummy_inputs
