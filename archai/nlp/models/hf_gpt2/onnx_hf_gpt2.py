# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Huggingface's Open AI GPT-2 for ONNX.
"""

from collections import OrderedDict
from typing import Any, Dict, Mapping

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
    def mockups(self) -> Mapping[str, torch.Tensor]:
        input_ids = torch.randint(0, self.config.n_token, (self.batch_size, self.seq_len))

        # Shape of past states
        # [past_key_values, batch_size, n_head, past_seq_len, d_head]
        past_key_values =  tuple([torch.zeros(self.config.past_key_values, self.batch_size, self.config.n_head, self.seq_len, self.config.d_head) for _ in range(self.config.n_layer)])

        return OrderedDict({'input_ids': input_ids, 'past_key_values': past_key_values})

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        input_ids = [('input_ids', {0: 'batch_size', 1: 'seq_len'})]

        # Shape of past states
        # [past_key_values, batch_size, n_head, past_seq_len, d_head]
        past_key_values = [(f'past_{i}', {1: 'batch_size', 3: 'past_seq_len'}) for i in range(self.config.n_layer)]

        return OrderedDict(input_ids + past_key_values)

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        probs = [('probs', {0: 'batch_size'})]

        # Shape of present states (past states when outputting)
        # [past_key_values, batch_size, n_head, total_seq_len, d_head]
        # Note that total_seq_len is seq_len + past_seq_len
        present_key_values = [(f'present_{i}', {1: 'batch_size', 3: 'total_seq_len'}) for i in range(self.config.n_layer)]

        return OrderedDict(probs + present_key_values)
