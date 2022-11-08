# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""GPT-2 ONNX configuration.
"""

from typing import Optional

from transformers.configuration_utils import PretrainedConfig

from archai.nlp.onnx.config_utils.onnx_config_base import OnnxConfigWithPast


class GPT2OnnxConfig(OnnxConfigWithPast):
    """Implements a GPT-2 ONNX configuration (with past key/values support)."""

    def __init__(
        self, config: PretrainedConfig, task: Optional[str] = "causal-lm", use_past: Optional[bool] = False
    ) -> None:
        super().__init__(config, task=task, use_past=use_past, past_key_values=2)

    @property
    def num_layers(self) -> int:
        return self.config.n_layer


class GPT2FlexOnnxConfig(GPT2OnnxConfig):
    """Implements a GPT-2 ONNX configuration (with past key/values support)."""

    def __init__(
        self, config: PretrainedConfig, task: Optional[str] = "causal-lm", use_past: Optional[bool] = False
    ) -> None:
        super().__init__(config, task=task, use_past=use_past)

    @property
    def num_attention_heads(self) -> int:
        return self.config.num_attention_heads[0]
