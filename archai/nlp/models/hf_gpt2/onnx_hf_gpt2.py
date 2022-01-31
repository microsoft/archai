# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Huggingface's Open AI GPT-2 for ONNX.
"""

from typing import Any, Dict

import torch
from onnxruntime.transformers.onnx_model_gpt2 import Gpt2OnnxModel as HfGPT2OnnxModel

from archai.nlp.models.config_base import BATCH_SIZE, SEQ_LEN, OnnxConfig


class HfGPT2OnnxConfig(OnnxConfig):
    def __init__(self, model_config: str) -> None:
        super().__init__(model_config)

        self.config['past_key_values'] = 2
        self.config['model_type'] = 'gpt2'

    @property
    def mockups(self) -> Dict[str, Any]:
        return {
            'input_ids': torch.randint(0, self.config['n_token'], (BATCH_SIZE, SEQ_LEN)),
            'past_key_values': tuple([torch.zeros(self.config['past_key_values'], BATCH_SIZE, self.config['n_head'], SEQ_LEN, self.config['d_head']) for _ in range(self.config['n_layer'])])
        }
