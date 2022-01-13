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
            'd_embed': 512,
            'dropout': 0.1,
            'dropatt': 0.0,
            'tgt_len': 192,
            'n_token': 267736,
            'weight_init_std': 0.0,
            'tie_weight': True,
            'primer_square': False
        }
