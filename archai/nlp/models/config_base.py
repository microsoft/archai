# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Configuration objects, such as ONNX.
"""

from collections import OrderedDict
from typing import Any, Dict

import torch

# ONNX-export constants
BATCH_SIZE = 1
SEQ_LEN = 32


class Config:
    def __init__(self) -> None:
        pass

    @property
    def default(self) -> Dict[str, Any]:
        return {}

    @property
    def search(self) -> Dict[str, Any]:
        return {}


class OnnxConfig:
    def __init__(self, model_config: str) -> None:
        self.config = model_config

    @property
    def mockups(self) -> Dict[str, Any]:
        return {
            'input_ids': torch.randint(0, self.config['n_token'], (BATCH_SIZE, SEQ_LEN)),
        }

    @property
    def inputs(self) -> OrderedDict:
        # Shape of past states
        # [past_key_values, batch_size, n_head, past_seq_len, d_head]
        pasts = [(f'past_{i}', {1: 'batch_size', 3: 'past_seq_len'}) for i in range(self.config['n_layer'])]
        return OrderedDict([('input_ids', {0: 'batch_size', 1: 'seq_len'})] + pasts)

    @property
    def outputs(self) -> OrderedDict:
        # Shape of present states (past states when outputting)
        # [past_key_values, batch_size, n_head, total_seq_len, d_head]
        # Note total_seq_len is current seq_len + past_seq_len
        presents = [(f'present_{i}', {1: 'batch_size', 3: 'total_seq_len'}) for i in range(self.config['n_layer'])]
        return OrderedDict([('probs', {0: 'batch_size'})] + presents)
