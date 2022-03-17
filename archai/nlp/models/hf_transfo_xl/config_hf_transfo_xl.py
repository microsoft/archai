# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Huggingface's Transformer-XL configurations.
"""

from typing import List, Optional

from archai.nlp.models.config_base import Config, SearchConfigParameter, SearchConfig
from transformers import CONFIG_MAPPING, PretrainedConfig


class HfTransfoXLConfig(Config, PretrainedConfig):
    """Huggingface's Transformer-XL default configuration.

    """

    hyperparameter_map = {
        'n_token': 'vocab_size',
        'weight_init_type': 'init',
        'weight_init_range': 'init_range',
        'weight_init_std': 'init_std',
    }
    hyperparameter_map.update(CONFIG_MAPPING['transfo-xl']().attribute_map)

    def __init__(self,
                 n_token: Optional[int] = 267736,
                 tgt_len: Optional[int] = 192,
                 d_model: Optional[int] = 512,
                 d_inner: Optional[int] = 2048,
                 d_head: Optional[int] = 0,
                 d_embed: Optional[int] = 0,
                 n_layer: Optional[int] = 16,
                 n_head: Optional[int] = 8,
                 dropout: Optional[float] = 0.1,
                 dropatt: Optional[float] = 0.0,
                 div_val: Optional[int] = 4,
                 pre_lnorm: Optional[bool] = False,
                 cutoffs: Optional[List[int]] = [19997, 39997, 199997],
                 mem_len: Optional[int] = 192,
                 same_length: Optional[bool] = False,
                 attn_type: Optional[int] = 0,
                 clamp_len: Optional[int] = -1,
                 sample_softmax: Optional[int] = -1,
                 adaptive: Optional[bool] = True,
                 weight_init_type: Optional[str] = 'normal',
                 weight_init_range: Optional[float] = 0.01,
                 weight_init_std: Optional[float] = 0.02,
                 proj_init_std: Optional[float] = 0.01,
                 tie_weight: Optional[bool] = True,
                 **kwargs) -> None:
        self.n_token = n_token
        self.tgt_len = tgt_len
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_head = d_head if d_head > 0 else d_model // n_head
        self.d_embed = d_embed if d_embed > 0 else d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.dropatt = dropatt
        self.div_val = div_val
        self.pre_lnorm = pre_lnorm
        self.cutoffs = cutoffs
        self.mem_len = mem_len
        self.same_length = same_length
        self.attn_type = attn_type
        self.clamp_len = clamp_len
        self.sample_softmax = sample_softmax
        self.adaptive = adaptive
        self.weight_init_type = weight_init_type
        self.weight_init_range = weight_init_range
        self.weight_init_std = weight_init_std
        self.proj_init_std = proj_init_std
        self.tie_weight = tie_weight

        additional_config = CONFIG_MAPPING['transfo-xl']().to_dict()
        for key, value in additional_config.items():
            if key not in self.__dict__.keys():
                setattr(self, key, value)

        super().__init__(**kwargs)


class HfTransfoXLSearchConfig(SearchConfig):
    """Huggingface's Transformer-XL search configuration.

    """

    def __init__(self) -> None:
        # Default HfTransfoXL search options: n_layer, d_model, d_inner and n_head
        n_layer = SearchConfigParameter(per_layer=False, value=[3, 4, 5, 6, 7, 8, 9, 10])
        d_model = SearchConfigParameter(per_layer=False, value=list(range(128, 1024, 64)))
        d_inner = SearchConfigParameter(per_layer=False, value=list(range(128, 4096, 64)))
        n_head = SearchConfigParameter(per_layer=False, value=[2, 4, 8])

        super().__init__(n_layer=n_layer,
                         d_model=d_model,
                         d_inner=d_inner,
                         n_head=n_head)
