# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""GPT-2 ONNX configuration.
"""

from typing import Any, Dict

from archai.nlp.onnx.config_utils.onnx_config_base import OnnxConfigWithPast


class GPT2OnnxConfig(OnnxConfigWithPast):
    def __init__(self, config, task="causal-lm", use_past=False) -> None:
        super().__init__(config, task=task, use_past=use_past, past_key_values=2)

    @property
    def num_layers(self) -> int:
        return self.config.n_layer