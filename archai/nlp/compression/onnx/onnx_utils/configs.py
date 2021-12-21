# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Configuration objects needed by ONNX when performing
export, quantization or any sort of operation.
"""

from collections import OrderedDict
from typing import Any, Dict

import torch

# Mockup constants
BATCH_SIZE = 1
SEQ_LEN = 32


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

        """

        return {
            'input_ids': torch.randint(0, self.config['n_token'], (BATCH_SIZE, SEQ_LEN)),
        }

    @property
    def inputs(self) -> OrderedDict:
        """Defines the inputs and their shapes to be used when exporting to ONNX.
        
        """

        # Shape of past states
        # [past_key_values, batch_size, n_head, past_seq_len, d_head]
        pasts = [(f'past_{i}', {1: 'batch_size', 3: 'past_seq_len'}) for i in range(self.config['n_layer'])]
        return OrderedDict([('input_ids', {0: 'batch_size', 1: 'seq_len'})] + pasts)

    @property
    def outputs(self) -> OrderedDict:
        """Defines the outputs and their shapes to be used when exporting to ONNX.
        
        """

        # Shape of present states (past states when outputting)
        # [2, batch_size, n_head, total_seq_len, d_head]
        # Note total_seq_len is current seq_len + past_seq_len
        presents = [(f'present_{i}', {1: 'batch_size', 3: 'total_seq_len'}) for i in range(self.config['n_layer'])]
        return OrderedDict([('probs', {0: 'batch_size'})] + presents)


class HfGPT2OnnxConfig(OnnxConfig):
    """Provides an ONNX-export configuration for HfGPT2.

    """

    def __init__(self, model_config: str) -> None:
        """Initializes the configuration.

        Args:
            model_config: Model configuration.

        """

        super().__init__(model_config)

        self.config['past_key_values'] = 2
        self.config['model_type'] = 'gpt2'

    @property
    def mockups(self) -> Dict[str, Any]:
        """Defines the mockups (inputs) to be used when exporting to ONNX.
        
        """

        return {
            'input_ids': torch.randint(0, self.config['n_token'], (BATCH_SIZE, SEQ_LEN)),
            'past_key_values': tuple([torch.zeros(self.config['past_key_values'], BATCH_SIZE, self.config['n_head'], SEQ_LEN, self.config['d_head']) for _ in range(self.config['n_layer'])])
        }


class MemTransformerLMOnnxConfig(OnnxConfig):
    """Provides an ONNX-export configuration for MemTransformerLM.

    """

    def __init__(self, model_config: str) -> None:
        """Initializes the configuration.

        Args:
            model_config: Model configuration.

        """

        super().__init__(model_config)

        # Checks the type of attention to define the `past_key_values`
        if self.config['attn_type'] == 0:
            # `k`, `v` and relative embeddings
            self.config['past_key_values'] = 3
        else:
            # `k` and `v`
            self.config['past_key_values'] = 2

        self.config['model_type'] = 'transfo-xl'

    @property
    def mockups(self) -> Dict[str, Any]:
        """Defines the mockups (inputs) to be used when exporting to ONNX.
        
        """

        return {
            'input_ids': torch.randint(0, self.config['n_token'], (BATCH_SIZE, SEQ_LEN)),
            'past_key_values': tuple([torch.zeros(self.config['past_key_values'], BATCH_SIZE, self.config['n_head'], SEQ_LEN, self.config['d_head']) for _ in range(self.config['n_layer'])])
        }
