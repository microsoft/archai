# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""ONNX-based configuration.
"""

from collections import OrderedDict
import copy
from typing import Mapping, Optional

import torch


class OnnxConfig:
    """"""

    DEFAULT_BATCH_SIZE = 2
    DEFAULT_SEQ_LEN = 8
    DEFAULT_TASK_OUTPUTS = {
        "causal-lm": OrderedDict({"probs": {0: "batch_size"}})
    }


    def __init__(self,
                 config,
                 task: Optional[str] = "causal-lm",
    ) -> None:
        """"""

        assert task in self.DEFAULT_TASK_OUTPUTS.keys(), f"`task`: {task} is not supported yet."

        self.config = config
        self.task = task

    @property
    def batch_size(self) -> int:
        """"""

        return self.DEFAULT_BATCH_SIZE

    @property
    def seq_len(self) -> int:
        """"""

        return self.DEFAULT_SEQ_LEN

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        """"""

        return OrderedDict({"input_ids": {0: "batch_size", 1: "seq_len"}})

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        """"""

        return copy.deepcopy(self.DEFAULT_TASK_OUTPUTS[self.task])

    def generate_dummy_inputs(self):
        """"""
        
        return {"input_ids": torch.zeros((self.batch_size, self.seq_len), dtype=torch.long)}


class OnnxConfigWithPast(OnnxConfig):
    """"""

    def __init__(self, config, task: Optional[str] = "causal-lm", use_past: Optional[bool] = False, past_key_values: Optional[int] = 2) -> None:
        super().__init__(config, task=task)

        if use_past:
            self.config.use_cache = True
            self.config.past_key_values = past_key_values
        else:
            self.config.use_cache = False

        self.use_past = use_past

    @property
    def hidden_size(self) -> int:
        if not hasattr(self.config, "hidden_size"):
            raise AttributeError()

        return self.config.hidden_size

    @property
    def num_layers(self) -> int:
        if not hasattr(self.config, "num_layers"):
            raise AttributeError()

        return self.config.num_layers

    @property
    def num_attention_heads(self) -> int:
        if not hasattr(self.config, "num_attention_heads"):
            raise AttributeError()

        return self.config.num_attention_heads

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        inputs = super().inputs

        if self.use_past:
            for i in range(self.num_layers):
                # [past_key_values, batch_size, n_head, past_seq_len, d_head]
                inputs[f"past_{i}"] = {1: "batch_size", 3: "past_seq_len"}

        return inputs

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        outputs = super().outputs

        if self.use_past:
            for i in range(self.num_layers):
                # [past_key_values, batch_size, n_head, total_seq_len, d_head]
                # Note that total_seq_len is seq_len + past_seq_len
                outputs[f"present_{i}"] = {1: "batch_size", 3: "total_seq_len"}

        return outputs

    def generate_dummy_inputs(self):
        """"""

        dummy_inputs = super().generate_dummy_inputs()

        if self.use_past:
            # [past_key_values, batch_size, n_head, past_seq_len, d_head]
            dummy_inputs["past_key_values"] = tuple(
                [
                    torch.zeros(self.config.past_key_values, self.batch_size, self.num_attention_heads, self.seq_len, self.hidden_size // self.num_attention_heads)
                    for _ in range(self.num_layers)
                ]
            )

        return dummy_inputs
