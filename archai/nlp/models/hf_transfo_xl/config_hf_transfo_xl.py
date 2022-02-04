# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Hugginface's Transformer-XL configuration.
"""

from typing import Any, Dict

from archai.nlp.models.config_base import Config


class HfTransfoXLConfig(Config):
    """Provides a configuration for HfTransfoXL.

    """

    def __init__(self, **kwargs) -> None:
        """Initializes the configuration.

        """

        super().__init__(**kwargs)

    @property
    def default(self) -> Dict[str, Any]:
        """Defines the default configuration used by the class.

        """

        return {
            'd_head': None,
            'd_embed': 512,
            'dropout': 0.1,
            'dropatt': 0.0,
            'n_token': 267736,
            'div_val': 4,
            'pre_lnorm': False,
            'cutoffs': [19997, 39997, 199997],
            'tgt_len': 192,
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
        """Defines the default configuration used when searching with the class.

        """

        return {
            'n_layer': {
                'per_layer': False,
                'value': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            },
            'd_model': {
                'per_layer': False,
                'value': [128, 256, 512, 768, 1024]
            },
            'd_inner': {
                'per_layer': False,
                'value': list(range(512, 2049, 50)) + list(range(2048, 3072, 200))
            },
            'n_head': {
                'per_layer': False,
                'value': [2, 4, 8]
            }
        }
