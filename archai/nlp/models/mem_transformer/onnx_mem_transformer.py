# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""MemTransformerLM ONNX-related classes and methods.
"""

from typing import Any, Dict

import torch

from archai.nlp.compression.onnx.onnx_utils.configs import OnnxConfig, BATCH_SIZE, SEQ_LEN


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
