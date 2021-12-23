# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Hugginface's Transformer-XL for ONNX.
"""

from typing import Any, Dict

import torch

from archai.nlp.common.constants import BATCH_SIZE, SEQ_LEN
from archai.nlp.models.config_base import OnnxConfig
from archai.nlp.models.mem_transformer.onnx_mem_transformer import \
    MemTransformerLMOnnxModel as HfTransfoXLOnnxModel


class HfTransfoXLOnnxConfig(OnnxConfig):
    """Provides an ONNX-export configuration for HfTransfoXL.

    """

    def __init__(self, model_config: str) -> None:
        """Initializes the configuration.

        Args:
            model_config: Model configuration.

        """

        super().__init__(model_config)

        self.config['model_type'] = 'transfo-xl'

    @property
    def mockups(self) -> Dict[str, Any]:
        """Defines the mockups (inputs) to be used when exporting to ONNX.
        
        """

        return {
            'input_ids': torch.randint(0, self.config['n_token'], (BATCH_SIZE, SEQ_LEN))
        }
