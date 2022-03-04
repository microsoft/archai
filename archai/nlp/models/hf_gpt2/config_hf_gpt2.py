# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Huggingface's Open AI GPT-2.
"""

from typing import Any, Dict

from archai.nlp.models.config_base import Config


class HfGPT2Config(Config):
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
            'tgt_len': 192,
            'n_token': 10000,
            'weight_init_std': 0.0,
            'tie_weight': True
        }

    @property
    def search(self) -> Dict[str, Any]:
        return {
            'n_layer': {
                'per_layer': False,
                'value': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            },
            'd_model': {
                'per_layer': False,
                'value': list(range(128, 1024, 64))
            },
            'd_inner': {
                'per_layer': False,
                'value': list(range(128, 4096, 64))
            },
            'n_head': {
                'per_layer': False,
                'value': [2, 4, 8]
            }
        }

        super().__init__(**kwargs)

    @property
    def default(self) -> Dict[str, Any]:
        """Defines the default configuration used by the class.

        """

        return {
            'd_head': None,
            'd_embed': -1,
            'dropout': 0.1,
            'dropatt': 0.0,
            'tgt_len': 192,
            'n_token': 10000,
            'weight_init_std': 0.0,
            'tie_weight': True,
            'primer_square': False
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
                'value': list(range(128, 1024, 64))
            },
            'd_inner': {
                'per_layer': True,
                'value': list(range(128, 4096, 64))
            },
            'n_head': {
                'per_layer': False,
                'value': [2, 4, 8]
            }
        }
