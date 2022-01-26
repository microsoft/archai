# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import torch
import torch.nn.functional as F
from transformers import CONFIG_MAPPING

from archai.nlp.models.archai_model import ArchaiModel
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel


class HfGPT2(ArchaiModel):
    """Adapts HuggingFace GPT2 model (GPT2LMHeadModel) to the transformer_xl codebase.
    """

    hyperparam_mapping = {'n_layer': 'n_layer',
                          'n_head': 'n_head',
                          'd_head': 'd_head',
                          'd_embed': 'n_embd',
                          'd_model': 'n_embd',
                          'd_inner': 'n_inner',
                          'dropout': 'resid_pdrop',
                          'dropatt': 'attn_pdrop',
                          'tgt_len': 'n_positions',
                          'n_token': 'vocab_size',
                          'weight_init_std': 'initializer_range'}

    def __init__(self, **kwargs) -> None:
        super(HfGPT2, self).__init__()

        assert len(kwargs['d_inner']) == kwargs['n_layer'] and len(kwargs['n_head']) == kwargs['n_layer'] and len(kwargs['d_head']) == kwargs['n_layer']
        assert all([n_head * d_head == kwargs['d_embed'] for n_head, d_head in zip(kwargs['n_head'], kwargs['d_head'])]), 'GPT2 does not support n_head*d_head != d_embed'
        assert kwargs['d_model'] == kwargs['d_embed'], 'GPT2 does not support d_model != d_embed'

        assert all(kwargs['d_inner'][0] == d_inner for d_inner in kwargs['d_inner']), 'GPT2 does not support heterogenous arch.'
        assert all(kwargs['d_head'][0] == d_head for d_head in kwargs['d_head']), 'GPT2 does not support heterogenous arch.'
        assert all(kwargs['n_head'][0] == n_head for n_head in kwargs['n_head']), 'GPT2 does not support heterogenous arch.'

        kwargs['d_inner'] = kwargs['d_inner'][0]
        kwargs['d_head'] = kwargs['d_head'][0]
        kwargs['n_head'] = kwargs['n_head'][0]

        # Translate hyperparams into HuggingFace GPT2 params
        self.config = self._generate_config(**kwargs)
        # Create model
        self.model = GPT2LMHeadModel(self.config)

        if kwargs['tie_weight']:
            self.model.tie_weights()

    def _generate_config(self, **kwargs):

        config = CONFIG_MAPPING['gpt2']()

        for param, gpt2_param in HfGPT2.hyperparam_mapping.items():
            setattr(config, gpt2_param, kwargs[param])

        config.embd_pdrop = kwargs['dropatt']

        return config

    def forward(self, input_ids:torch.Tensor, labels:Optional[torch.Tensor], mems:Optional[torch.Tensor],
                past_key_values:Optional[torch.Tensor]=None, output_loss=True, output_prediction_scores=False):

        assert mems is None, 'GPT2 does not support memory (mems)'

        # Labels in GPT2LMHeadModel are the same as inputs, the offset between inputs annd labels is done
        # inside the model. The causal attention mask is also created inside the model.
        hf_out = self.model(input_ids=input_ids, labels=input_ids, attention_mask=torch.ones_like(input_ids))

        # GPT2LMHeadModel only outputs logits, so we convert it using log_softmax
        return (hf_out.loss, F.log_softmax(hf_out.logits, dim=-1), None, past_key_values)

    def reset_length(self, tgt_len: int, ext_len: int, mem_len: int):
        # We don't have memory as in transformer xl
        pass

    def get_non_emb_params(self):
        return sum([p.nelement() for p in self.model.transformer.h.parameters()])
