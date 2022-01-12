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
    """Provides a foundation for creating configuration instances.

    """

    def __init__(self, **kwargs) -> None:
        """Initializes the configuration and sets keywords as attributes.

        """

        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def default(self) -> Dict[str, Any]:
        """Defines the default configuration used by the class.

        """

        return {}


class OnnxConfig:
    """Provides a foundation for creating ONNX-export configuration instances.

    """

    def __init__(self, model_config: str) -> None:
        """Initializes the configuration.

        Args:
            model_config: Model configuration.

        """

        self.config = model_config

    @property
    def mockups(self) -> Dict[str, Any]:
        """Defines the mockups (inputs) to be used when exporting to ONNX.

        Returns:
            (Dict[str, Any]): Mockups used to export with ONNX.

        """

        return {
            'input_ids': torch.randint(0, self.config['n_token'], (BATCH_SIZE, SEQ_LEN)),
        }

    @property
    def inputs(self) -> OrderedDict:
        """Defines the inputs and their shapes to be used when exporting to ONNX.

        Returns:
            (OrderedDict): Inputs used to export with ONNX.
        
        """

        # Shape of past states
        # [past_key_values, batch_size, n_head, past_seq_len, d_head]
        pasts = [(f'past_{i}', {1: 'batch_size', 3: 'past_seq_len'}) for i in range(self.config['n_layer'])]
        return OrderedDict([('input_ids', {0: 'batch_size', 1: 'seq_len'})] + pasts)

    @property
    def outputs(self) -> OrderedDict:
        """Defines the outputs and their shapes to be used when exporting to ONNX.

        Returns:
            (OrderedDict): Outputs used to export with ONNX.
        
        """

        # Shape of present states (past states when outputting)
        # [2, batch_size, n_head, total_seq_len, d_head]
        # Note total_seq_len is current seq_len + past_seq_len
        presents = [(f'present_{i}', {1: 'batch_size', 3: 'total_seq_len'}) for i in range(self.config['n_layer'])]
        return OrderedDict([('probs', {0: 'batch_size'})] + presents)
