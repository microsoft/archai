# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""NVIDIA's Memory Transformer (Transformer-XL) configuration.
"""

from typing import Any, Dict
<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> b64935ed (chore(nlp): Adds configuration classes to every available model.)
=======

>>>>>>> 929e81d1 (chore(nlp): Removing the need of model configuration defaults on NAS.)
from archai.nlp.models.config_base import Config


class MemTransformerLMConfig(Config):
<<<<<<< HEAD
    @property
    def default(self) -> Dict[str, Any]:
        return {
            'd_head': -1,
            'n_token': 267736,
            'dropout': 0.1,
            'dropatt': 0.0,
            'd_embed': -1,
=======
    """Provides a configuration for MemTransformerLM.

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
            'n_token': 267736,
            'dropout': 0.1,
            'dropatt': 0.0,
            'd_embed': None,
>>>>>>> b64935ed (chore(nlp): Adds configuration classes to every available model.)
            'div_val': 4,
            'pre_lnorm': False,
            'tgt_len': 192,
            'ext_len': 0,
            'mem_len': 192,
            'same_length': False,
            'attn_type': 0,
            'clamp_len': -1,
            'sample_softmax': -1,
            'cutoffs': [19997, 39997, 199997],
            'tie_projs': [False, True, True, True],
            'tie_weight': True,
            'dtype': None,
            'primer_conv': False,
            'primer_square': False,
            'use_cache': False
        }
<<<<<<< HEAD

    @property
    def search(self) -> Dict[str, Any]:
        return {
            'n_layer': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'd_model': [128, 256, 512, 768, 1024],
            'd_inner': list(range(512, 2049, 50)) + list(range(2048, 3072, 200)),
            'n_head': [2, 4, 8]
        }
=======
>>>>>>> b64935ed (chore(nlp): Adds configuration classes to every available model.)
