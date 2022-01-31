# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Huggingface's Open AI GPT-2.
"""

from typing import Any, Dict

from archai.nlp.models.config_base import Config


class HfGPT2Config(Config):
    """Provides a configuration for HfGPT2.

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
            'n_token': 267736,
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


class HfGPT2FlexConfig(Config):
    """Provides a configuration for HfGPT2Flex.

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
            'n_token': 267736,
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
            'n_head': [2, 4, 8]
        }
