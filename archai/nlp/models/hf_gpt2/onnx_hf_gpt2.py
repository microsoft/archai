# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from typing import Dict, Any

from onnxruntime.transformers.onnx_model_gpt2 import Gpt2OnnxModel as HfGPT2OnnxModel
from archai.nlp.models.config_base import OnnxConfig

import archai.nlp.common.constants as c

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
            'input_ids': torch.randint(0, self.config['n_token'], (c.BATCH_SIZE, c.SEQ_LEN)),
            'past_key_values': tuple([torch.zeros(self.config['past_key_values'], c.BATCH_SIZE, self.config['n_head'], c.SEQ_LEN, self.config['d_head']) for _ in range(self.config['n_layer'])])
        }
