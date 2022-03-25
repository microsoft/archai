# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Huggingface's Open AI GPT-2 for ONNX.
"""

from typing import Any, Dict

import torch
from onnxruntime.transformers.onnx_model_gpt2 import Gpt2OnnxModel as HfGPT2OnnxModel

from archai.nlp.models.config_base import OnnxConfig


class HfGPT2OnnxConfig(OnnxConfig):
    """Huggingface's Open AI GPT-2 ONNX-based configuration.

    """

    def __init__(self, model_config: Dict[str, Any]) -> None:
        """Initializes the class by setting missing keys on incoming
            model's configuration.

        Args:
            model_config: Configuration of the model that will be exported.

        """

        model_config['past_key_values'] = 2
        model_config['model_type'] = 'gpt2'

        super().__init__(model_config)

    @property
    def mockups(self) -> Dict[str, Any]:
        return {
            'input_ids': torch.randint(0, self.config.n_token, (self.batch_size, self.seq_len)),
            'past_key_values': tuple([torch.zeros(self.config.past_key_values, self.batch_size, self.config.n_head, self.seq_len, self.config.d_head) for _ in range(self.config.n_layer)])
        }
