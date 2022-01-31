# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Hugginface's Transformer-XL for ONNX.
"""

from typing import Any, Dict

import torch

from archai.nlp.models.config_base import BATCH_SIZE, SEQ_LEN, OnnxConfig
from archai.nlp.models.mem_transformer.onnx_mem_transformer import MemTransformerLMOnnxModel as HfTransfoXLOnnxModel


class HfTransfoXLOnnxConfig(OnnxConfig):
    def __init__(self, model_config: str) -> None:
        super().__init__(model_config)

        self.config['model_type'] = 'transfo-xl'

    @property
    def mockups(self) -> Dict[str, Any]:
        return {
            'input_ids': torch.randint(0, self.config['n_token'], (BATCH_SIZE, SEQ_LEN))
        }
