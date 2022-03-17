# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Huggingface's Open AI GPT-2.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

from archai.nlp.models.model_base import ArchaiModel
from archai.nlp.models.hf_gpt2.config_hf_gpt2_ import HfGPT2Config

class HfGPT2(ArchaiModel):
    """
    """

    def __init__(self, **kwargs) -> None:
        super(HfGPT2, self).__init__()

        self.config = HfGPT2Config(**kwargs)
        self.model = GPT2LMHeadModel(self.config)

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
        assert mems is None, 'GPT2 does not support memory (mems)'

        # Labels in Huggingface's GPT-2 are the same as inputs_ids and they will be shifted inside the model
        # Causal attention mask is created inside the model
        hf_out = self.model(input_ids=input_ids,
                            labels=input_ids,
                            attention_mask=torch.ones_like(input_ids))

        # GPT-2 only outputs the logits, so we need to convert them
        # by using log softmax
        return (hf_out.loss, F.log_softmax(hf_out.logits, dim=-1), None, past_key_values)

    def reset_length(self, tgt_len: int, ext_len: int, mem_len: int) -> None:
        # There is no memory in GPT-2
        pass

    def get_params(self) -> Dict[str, int]:
        params = {}

        params['embedding'] = self.get_params_from_layer(['Embedding'])
        params['attention'] = self.get_params_from_layer(['GPT2Attention'])
        params['ff'] = self.get_params_from_layer(['GPT2MLP'])
        params['layer_norm'] = self.get_params_from_layer(['LayerNorm'])

        params['non_embedding'] = params['attention'] + params['ff'] + params['layer_norm']
        params['total'] = params['non_embedding'] + params['embedding']

        return params
