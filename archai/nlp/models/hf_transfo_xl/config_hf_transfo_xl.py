# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Hugginface's Transformer-XL configuration.
"""

from typing import Any, Dict

from archai.nlp.models.config_base import Config


class HfTransfoXLConfig(Config):
    @property
    def default(self) -> Dict[str, Any]:
        return {
            'd_model': 512,
            'd_inner': 2048,
            'd_head': -1,
            'd_embed': -1,
            'n_layer': 16,
            'n_head': 8,
            'dropout': 0.1,
            'dropatt': 0.0,
            'n_token': 267736,
            'div_val': 4,
            'pre_lnorm': False,
            'cutoffs': [19997, 39997, 199997],
            'mem_len': 192,
            'same_length': False,
            'attn_type': 0,
            'clamp_len': -1,
            'sample_softmax': -1,
            'adaptive': True,
            'weight_init_type': 'normal',
            'weight_init_range': 0.01,
            'weight_init_std': 0.02,
            'proj_init_std': 0.01,
            'tie_weight': True
        }

    @property
    def search(self) -> Dict[str, Any]:
        return {
            'n_layer': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'd_model': [128, 256, 512, 768, 1024],
            'd_inner': list(range(512, 2049, 50)) + list(range(2048, 3072, 200)),
            'n_head': [2, 4, 8]
        }
