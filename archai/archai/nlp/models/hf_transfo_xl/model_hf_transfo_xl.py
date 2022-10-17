# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Hugginface's Transformer-XL.
"""

from typing import Dict, Optional, Tuple

import torch
from transformers import TransfoXLLMHeadModel

from archai.nlp.models.hf_transfo_xl.config_hf_transfo_xl import HfTransfoXLConfig
from archai.nlp.models.model_base import ArchaiModel


class HfTransfoXL(ArchaiModel):
    """Huggingface's Transformer-XL standard architecture.

    """
    
    def __init__(self, **kwargs) -> None:
        """Initializes the class by creating compatible configuration and model objects.
        
        """
        
        super().__init__()

        self.config = HfTransfoXLConfig(**kwargs)
        self.model = TransfoXLLMHeadModel(self.config)

        if self.config.tie_weight:
            self.model.tie_weights()

    def forward(self,
                input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                mems: Optional[torch.Tensor] = None,
                past_key_values: Optional[torch.Tensor] = None,
                output_loss: Optional[bool] = True,
                output_prediction_scores: Optional[bool] = False
                ) -> Tuple[torch.Tensor, ...]:
        # Labels are the same as input_ids because they will be shifted inside the model
        if output_loss:
            outputs = self.model(input_ids=input_ids,
                                 labels=input_ids,
                                 mems=mems)

            return (outputs.losses, None, outputs.mems, None)

        if output_prediction_scores:
            outputs = self.model(input_ids=input_ids,
                                 mems=mems)
                                
            return (None, outputs.logits, outputs.mems, None)

    def get_params(self) -> Dict[str, int]:
        params = {}

        params['embedding'] = self.get_params_from_layer(['AdaptiveEmbedding'])
        params['softmax'] = self.get_params_from_layer(['ProjectedAdaptiveLogSoftmax'])
        params['attention'] = self.get_params_from_layer(['RelPartialLearnableMultiHeadAttn'])
        params['ff'] = self.get_params_from_layer(['PositionwiseFF'])

        params['non_embedding'] = params['softmax'] + params['attention'] + params['ff']
        params['total'] = params['non_embedding'] + params['embedding']

        return params

    def reset_length(self, tgt_len: int, ext_len: int, mem_len: int) -> None:
        if tgt_len < 1:
            raise ValueError(f'tgt_len: {tgt_len} should be >= 1.')
        if ext_len < 0:
            raise ValueError(f'ext_len: {ext_len} should be >= 0.')
        if mem_len < 0:
            raise ValueError(f'mem_len: {mem_len} should be >= 0.')

        self.model.config.tgt_len = tgt_len
        self.model.config.mem_len = mem_len
        self.model.config.ext_len = ext_len
