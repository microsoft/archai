# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Hugginface's Transformer-XL.
"""

from typing import Dict, Optional, Tuple

import torch
from transformers import CONFIG_MAPPING, AutoModelForCausalLM

from archai.common.utils import map_to_list
from archai.nlp.models.model_base import ArchaiModel


class HfTransfoXL(ArchaiModel):
    """Huggingface's Transformer-XL.

    """

    HYPERPARAMETER_MAPPING = {
        'n_layer': 'n_layer',
        'n_head': 'n_head',
        'd_head': 'd_head',
        'd_embed': 'd_embed',
        'd_model': 'd_model',
        'd_inner': 'd_inner',
        'dropout': 'dropout',
        'dropatt': 'dropatt',
        'n_token': 'vocab_size',
        'div_val': 'div_val',
        'pre_lnorm': 'pre_lnorm',
        'cutoffs': 'cutoffs',
        'mem_len': 'mem_len',
        'same_length': 'same_length',
        'attn_type': 'attn_type',
        'clamp_len': 'clamp_len',
        'sample_softmax': 'sample_softmax',
        'adaptive': 'adaptive',
        'weight_init_type': 'init',
        'weight_init_range': 'init_range',
        'weight_init_std': 'init_std',
        'proj_init_std': 'proj_init_std'
    }

    def __init__(self, **kwargs) -> None:
        """Overrides initialization method.

        """

        super(HfTransfoXL, self).__init__()

        d_inner = map_to_list(kwargs['d_inner'], kwargs['n_layer'])
        n_head = map_to_list(kwargs['n_head'], kwargs['n_layer'])
        d_head = [kwargs['d_model'] // n_h for n_h in kwargs['n_head']] if kwargs['d_head'] is None else map_to_list(kwargs['d_head'], kwargs['n_layer'])

        assert len(d_inner) == kwargs['n_layer'] and len(n_head) == kwargs['n_layer'] and len(d_head) == kwargs['n_layer']

        kwargs['d_inner'] = d_inner[0]
        kwargs['n_head'] = n_head[0]
        kwargs['d_head'] = d_head[0]

        # Translate the hyperparameters into Huggingface's TransfoXL hyperparameters,
        # and creates the model with the proper configuration
        self.config = self._generate_config(**kwargs)
        self.model = AutoModelForCausalLM.from_config(self.config)

        if kwargs['tie_weight']:
            self.model.tie_weights()

    def _generate_config(self, **kwargs) -> None:
        """Generates a proper configuration according to mapped hyperparameters.

        """

        config = CONFIG_MAPPING['transfo-xl']()

        for param, transfo_xl_param in HfTransfoXL.HYPERPARAMETER_MAPPING.items():
            setattr(config, transfo_xl_param, kwargs[param])

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

        # Labels in Huggingface's TransfoXL are the same as inputs_ids,
        # and they will be shifted inside the model
        if output_loss:
            hf_out = self.model(input_ids=input_ids,
                                labels=input_ids,
                                mems=mems)

            return (hf_out.losses, None, hf_out.mems, past_key_values)

        if output_prediction_scores:
            hf_out = self.model(input_ids=input_ids,
                                mems=mems)
                                
            return (None, hf_out.logits, hf_out.mems, past_key_values)

    def reset_length(self, tgt_len: int, ext_len: int, mem_len: int) -> None:
        """Resets the length of the memory.

        Args:
            tgt_len: Length of target sample.
            ext_len: Length of extended memory.
            mem_len: Length of the memory.

        """

        if tgt_len < 1:
            raise RuntimeError(f'tgt_len should be >= 1, but got {tgt_len}')
        if ext_len < 0:
            raise RuntimeError(f'ext_len should be >= 0, but got {ext_len}')
        if mem_len < 0:
            raise RuntimeError(f'mem_len should be >= 0, but got {mem_len}')

        self.model.config.tgt_len = tgt_len
        self.model.config.mem_len = mem_len
        self.model.config.ext_len = ext_len

    def get_params(self) -> Dict[str, int]:
        """Returns a dictionary of total parameters per implemented layer.

        Returns:
            (Dict[str, int]): Number of total parameters.

        """

        params = {}

        params['embedding'] = self.get_params_from_layer(['AdaptiveEmbedding'])
        params['softmax'] = self.get_params_from_layer(['ProjectedAdaptiveLogSoftmax'])
        params['attention'] = self.get_params_from_layer(['RelPartialLearnableMultiHeadAttn'])
        params['ff'] = self.get_params_from_layer(['PositionwiseFF'])

        params['non_embedding'] = params['attention'] + params['ff']
        params['total'] = params['non_embedding'] + params['embedding'] + params['softmax']

        return params
