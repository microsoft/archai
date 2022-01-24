# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Huggingface's Open AI GPT-2.
"""

import types
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import CONFIG_MAPPING, AutoModelForCausalLM

from archai.nlp.common.mapping_utils import map_to_list
from archai.nlp.models.model_base import ArchaiModel
from archai.nlp.models.model_utils.primer_ez import forward_hf_gpt2_mlp_primer_ez


class HfGPT2(ArchaiModel):
    """Huggingface's Open AI GPT-2.

    """

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
        """Overrides initialization method.

        """

        super(HfGPT2, self).__init__()

        if kwargs['d_embed'] == -1:
            kwargs['d_embed'] = kwargs['d_model']

        d_inner = map_to_list(kwargs['d_inner'], kwargs['n_layer'])
        n_head = map_to_list(kwargs['n_head'], kwargs['n_layer'])
        d_head = [kwargs['d_model'] // n_h for n_h in kwargs['n_head']] if kwargs['d_head'] is None else map_to_list(kwargs['d_head'], kwargs['n_layer'])

        assert len(d_inner) == kwargs['n_layer'] and len(n_head) == kwargs['n_layer'] and len(d_head) == kwargs['n_layer']
        
        kwargs['d_inner'] = d_inner[0]
        kwargs['n_head'] = n_head[0]
        kwargs['d_head'] = d_head[0]

        assert kwargs['d_model'] == kwargs['d_embed'], 'GPT2 does not support d_model != d_embed'
        assert kwargs['n_head'] * kwargs['d_head'] == kwargs['d_embed'], 'GPT2 does not support n_head*d_head != d_embed'

        # Translate the hyperparameters into Huggingface's GPT-2 hyperparameters,
        # and creates the model with the proper configuration
        self.config = self._generate_config(**kwargs)
        self.model = AutoModelForCausalLM.from_config(self.config)

        if kwargs['tie_weight']:
            self.model.tie_weights()

        if kwargs['primer_square']:
            for block in self.model.transformer.h:
                block.mlp.forward = types.MethodType(forward_hf_gpt2_mlp_primer_ez, block.mlp)

    def _generate_config(self, **kwargs) -> None:
        """Generates a proper configuration according to mapped hyperparameters.

        """

        config = CONFIG_MAPPING['gpt2']()

        # Embedding dropout we always set to zero
        config.embd_pdrop = kwargs['dropatt']
        
        # Checks if it is supposed to use PrimerEZ squared ReLU
        if kwargs['primer_square']:
            config.activation_function = 'relu'

        for param, gpt2_param in HfGPT2.HYPERPARAMETER_MAPPING.items():
            setattr(config, gpt2_param, kwargs[param])

        return config

    def forward(self,
                input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                mems: Optional[torch.Tensor] = None,
                past_key_values: Optional[torch.Tensor] = None,
                output_loss: Optional[bool] = True,
                output_prediction_scores: Optional[bool] = False
                ) -> Tuple[torch.Tensor, ...]:
        """Performs forward pass over the model.

        Args:
            input_ids: Input tokens.
            labels: Input labels (same as tokens).
            mems: Memory tensor.
            past_key_values: Tensor with past key/values.
            output_loss: Whether loss should be outputted.
            output_prediction_scores: Whether prediction scores should be outputted.

        Returns:
            (Tuple[torch.Tensor, ...]): Outputs, such as loss, prediction scores,
                memories and past key/values.

        """

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
        """Resets the length of the memory.

        Args:
            tgt_len: Length of target sample.
            ext_len: Length of extended memory.
            mem_len: Length of the memory.

        """

        # There is no memory in GPT-2
        pass

    def get_params(self) -> Dict[str, int]:
        """Returns a dictionary of total parameters per implemented layer.

        Returns:
            (Dict[str, int]): Number of total parameters.

        """

        params = {}

        params['embedding'] = self.get_params_from_layer(['nn.Embedding'])
        params['attention'] = self.get_params_from_layer(['GPT2Attention'])
        params['ff'] = self.get_params_from_layer(['GPT2MLP'])

        params['non_embedding'] = params['attention'] + params['ff']
        params['total'] = params['non_embedding'] + params['embedding']

        return params
