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
            'vocab_size': 267736,
            'dropout': 0.1,
            'dropatt': 0.0,
            'd_embed': None,
            'div_val': 4,
            'pre_lnorm': False,
            'mem_len': 192,
            'same_length': False,
            'attn_type': 0,
            'clamp_len': -1,
            'sample_softmax': -1,
            'cutoffs': [19997, 39997, 199997]
        }
