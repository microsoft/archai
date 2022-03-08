# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Huggingface's Open AI GPT-2.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import CONFIG_MAPPING
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

from archai.common.utils import map_to_list
from archai.nlp.models.model_base import ArchaiModel
from archai.nlp.models.hf_gpt2.hf_gpt2_utils.gpt2_lm_head_model_flex import GPT2LMHeadModelFlex


class HfGPT2(ArchaiModel):
    HYPERPARAMETER_MAPPING = {
        'n_layer': 'n_layer',
        'n_head': 'n_head',
        'd_head': 'd_head',
        'd_embed': 'n_embd',
        'd_model': 'n_embd',
        'd_inner': 'n_inner',
        'dropout': 'resid_pdrop',
        'dropatt': 'attn_pdrop',
        'tgt_len': 'n_positions',
        'n_token': 'vocab_size',
        'weight_init_std': 'initializer_range'
    }

    def __init__(self, **kwargs) -> None:
        super(HfGPT2, self).__init__()

        if kwargs['d_embed'] < 0:
            kwargs['d_embed'] = kwargs['d_model']

        kwargs['d_inner'] = map_to_list(kwargs['d_inner'], kwargs['n_layer'])
        kwargs['n_head'] = map_to_list(kwargs['n_head'], kwargs['n_layer'])
        kwargs['d_head'] = [kwargs['d_model'] // n_h for n_h in kwargs['n_head']] if kwargs['d_head'] < 0 else map_to_list(kwargs['d_head'], kwargs['n_layer'])

        assert len(kwargs['d_inner']) == kwargs['n_layer'] and len(kwargs['n_head']) == kwargs['n_layer'] and len(kwargs['d_head']) == kwargs['n_layer']
        
        kwargs['d_inner'] = kwargs['d_inner'][0]
        kwargs['n_head'] = kwargs['n_head'][0]
        kwargs['d_head'] = kwargs['d_head'][0]

        assert kwargs['d_model'] == kwargs['d_embed'], 'GPT2 does not support d_model != d_embed.'
        assert kwargs['n_head'] * kwargs['d_head'] == kwargs['d_embed'], 'GPT2 does not support n_head * d_head != d_embed.'

        # Translate the hyperparameters into Huggingface's GPT-2 hyperparameters,
        # and creates the model with the proper configuration
        self.config = self._generate_config(**kwargs)
        self.model = GPT2LMHeadModel(self.config)

        if kwargs['tie_weight']:
            self.model.tie_weights()

    def _generate_config(self, **kwargs) -> None:
        config = CONFIG_MAPPING['gpt2']()

        for param, gpt2_param in HfGPT2.HYPERPARAMETER_MAPPING.items():
            setattr(config, gpt2_param, kwargs[param])

        config.embd_pdrop = kwargs['dropatt']

        return config

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

        params['embedding'] = self.get_params_from_layer(['nn.Embedding'])
        params['attention'] = self.get_params_from_layer(['GPT2Attention'])
        params['ff'] = self.get_params_from_layer(['GPT2MLP'])

        params['non_embedding'] = params['attention'] + params['ff']
        params['total'] = params['non_embedding'] + params['embedding']

        return params


class HfGPT2Flex(ArchaiModel):
    HYPERPARAMETER_MAPPING = {
        'n_layer': 'n_layer',
        'n_head': 'n_head',
        'd_head': 'd_head',
        'd_embed': 'n_embd',
        'd_model': 'n_embd',
        'd_inner': 'n_inner',
        'dropout': 'resid_pdrop',
        'dropatt': 'attn_pdrop',
        'tgt_len': 'n_positions',
        'n_token': 'vocab_size',
        'weight_init_std': 'initializer_range'
    }

    def __init__(self, **kwargs) -> None:
        super(HfGPT2Flex, self).__init__()

        if kwargs['d_embed'] < 0:
            kwargs['d_embed'] = kwargs['d_model']

        kwargs['d_inner'] = map_to_list(kwargs['d_inner'], kwargs['n_layer'])
        kwargs['n_head'] = map_to_list(kwargs['n_head'], kwargs['n_layer'])
        kwargs['d_head'] = [kwargs['d_model'] // n_h for n_h in kwargs['n_head']] if kwargs['d_head'] < 0 else map_to_list(kwargs['d_head'], kwargs['n_layer'])

        assert len(kwargs['d_inner']) == kwargs['n_layer'] and len(kwargs['n_head']) == kwargs['n_layer'] and len(kwargs['d_head']) == kwargs['n_layer']
        
        assert all([n_h * d_h == kwargs['d_embed'] for n_h, d_h in zip(kwargs['n_head'], kwargs['d_head'])]), 'GPT2 Flex does not support n_head * d_head != d_embed.'
        assert kwargs['d_model'] == kwargs['d_embed'], 'GPT2 flex does not support d_model != d_embed.'

        assert all(kwargs['n_head'][0] == n_h for n_h in kwargs['n_head']), 'GPT2 Flex does not support different attention heads.'
        assert all(kwargs['d_head'][0] == d_h for d_h in kwargs['d_head']), 'GPT2 Flex does not support different attention heads.'

        # Translate the hyperparameters into Huggingface's GPT-2 hyperparameters,
        # and creates the model with the proper configuration
        self.config = self._generate_config(**kwargs)
        self.model = GPT2LMHeadModelFlex(self.config)
        
        if kwargs['tie_weight']:
            self.model.tie_weights()

    def _generate_config(self, **kwargs):
        config = CONFIG_MAPPING['gpt2']()

        for param, gpt2_param in HfGPT2Flex.HYPERPARAMETER_MAPPING.items():
            setattr(config, gpt2_param, kwargs[param])

        if kwargs['primer_square']:
            config.primer_square = True
            config.activation_function = 'relu'
        else:
            config.primer_square = False

        config.embd_pdrop = kwargs['dropatt']

        return config

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
        params['ff'] = self.get_params_from_layer(['GPT2MLPFlex'])

        params['non_embedding'] = params['attention'] + params['ff']
        params['total'] = params['non_embedding'] + params['embedding']

        return params
