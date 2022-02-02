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
            'n_layer': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'd_model': [128, 256, 512, 768, 1024],
            'd_inner': list(range(512, 2049, 50)) + list(range(2048, 3072, 200)),
            'n_head': [8]
        }
