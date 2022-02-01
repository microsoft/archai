# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Hugginface's Transformer-XL for ONNX.
"""

from collections import OrderedDict

from archai.nlp.models.config_base import OnnxConfig
from archai.nlp.models.mem_transformer.onnx_mem_transformer import MemTransformerLMOnnxModel as HfTransfoXLOnnxModel


class HfTransfoXLOnnxConfig(OnnxConfig):
    def __init__(self, model_config: str) -> None:
        super().__init__(model_config)

        self.config['past_key_values'] = 0
        self.config['model_type'] = 'transfo-xl'

    @property
    def inputs(self) -> OrderedDict:
        return OrderedDict([('input_ids', {0: 'batch_size', 1: 'seq_len'})])

    @property
    def outputs(self) -> OrderedDict:
        return OrderedDict([('probs', {0: 'batch_size'})])
