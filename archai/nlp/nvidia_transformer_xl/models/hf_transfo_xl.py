from typing import Optional

import torch

from transformers import (
    CONFIG_MAPPING,
    AutoModelForCausalLM
)

from archai.nlp.nvidia_transformer_xl.models.model_utils import map_to_list
from archai.nlp.nvidia_transformer_xl.models.archai_model import ArchaiModel


class HfTransfoXL(ArchaiModel):
    """Adapts HuggingFace TransfoXL model (TransfoXLLMHeadModel) to the transformer_xl codebase.
    """

    hyperparam_mapping = {'n_layer': 'n_layer',
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
                          'proj_init_std': 'proj_init_std'}

    def __init__(self, **kwargs) -> None:
        super(HfTransfoXL, self).__init__()

        d_inner = map_to_list(kwargs['d_inner'], kwargs['n_layer'])
        n_head = map_to_list(kwargs['n_head'], kwargs['n_layer'])
        d_head = [kwargs['d_model'] // n_h for n_h in kwargs['n_head']] if kwargs['d_head'] is None else map_to_list(kwargs['d_head'], kwargs['n_layer'])

        assert len(d_inner) == kwargs['n_layer'] and len(n_head) == kwargs['n_layer'] and len(d_head) == kwargs['n_layer']

        kwargs['d_inner'] = d_inner[0]
        kwargs['n_head'] = n_head[0]
        kwargs['d_head'] = d_head[0]

        # Translate hyperparams into HuggingFace TransfoXL params
        self.config = self._generate_config(**kwargs)

        # Create model
        self.model = AutoModelForCausalLM.from_config(self.config)

        if kwargs['tie_weight']:
            self.model.tie_weights()

    def _generate_config(self, **kwargs):

        config = CONFIG_MAPPING['transfo-xl']()

        for param, transfo_xl_param in HfTransfoXL.hyperparam_mapping.items():
            setattr(config, transfo_xl_param, kwargs[param])

        return config

    def forward(self, input_ids:torch.Tensor, labels:Optional[torch.Tensor], mems:Optional[torch.Tensor],
                past_key_values:Optional[torch.Tensor]=None, output_loss=True, output_prediction_scores=False):
        # Labels in TransfoXLLMHeadModel are the same as inputs, the offset between inputs and labels is done
        # inside the model. The causal attention mask is also created inside the model.
        if output_loss:
            hf_out = self.model(input_ids=input_ids, labels=input_ids, mems=mems)
            return (hf_out.losses, None, hf_out.mems, past_key_values)

        if output_prediction_scores:
            hf_out = self.model(input_ids=input_ids, mems=mems)
            return (None, hf_out.logits, hf_out.mems, past_key_values)

    def reset_length(self, tgt_len: int, ext_len: int, mem_len: int):
        if tgt_len < 1:
            raise RuntimeError(f'tgt_len should be >= 1, but got {tgt_len}')
        if ext_len < 0:
            raise RuntimeError(f'ext_len should be >= 0, but got {ext_len}')
        if mem_len < 0:
            raise RuntimeError(f'mem_len should be >= 0, but got {mem_len}')
        self.model.config.tgt_len = tgt_len
        self.model.config.mem_len = mem_len
        self.model.config.ext_len = ext_len

    def get_non_emb_params(self):
        return sum([p.nelement() for p in self.model.transformer.layers.parameters()])
