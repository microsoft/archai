# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Optional, Tuple

from transformers.configuration_utils import PretrainedConfig

from archai.onnx.config_utils.onnx_config_base import OnnxConfigWithPast


class CodeGenOnnxConfig(OnnxConfigWithPast):
    """CodeGen ONNX configuration (with past key/values support)."""

    def __init__(
        self,
        config: PretrainedConfig,
        task: Optional[str] = "causal-lm",
        use_past: Optional[bool] = False,
    ) -> None:
        super().__init__(config, task=task, use_past=use_past, past_key_values=2)

    @property
    def num_layers(self) -> int:
        return self.config.n_layer

    @property
    def is_ort_graph_optimizable(self) -> bool:
        return False

    @property
    def ort_graph_optimizer_args(self) -> Tuple[Any, ...]:
        return (self.num_attention_heads, self.hidden_size)
