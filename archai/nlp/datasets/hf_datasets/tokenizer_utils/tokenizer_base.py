# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

from __future__ import annotations

import os
from typing import Optional

from tokenizers import Tokenizer
from tokenizers.trainers import Trainer

from archai.nlp.datasets.hf_datasets.tokenizer_utils.token_config import TokenConfig


class TokenizerBase:
    """
    """

    def __init__(
        self, tokenizer: Tokenizer, token_config: TokenConfig, trainer: Trainer
    ) -> None:
        """
        """

        self.tokenizer = tokenizer
        self.token_config = token_config
        self.trainer = trainer

    def train(self, files) -> None:
        """
        """

        return self.tokenizer.train(files, trainer=self.trainer)

    def train_from_iterator(self, iterator) -> None:
        """
        """
                
        return self.tokenizer.train_from_iterator(iterator, trainer=self.trainer, length=len(iterator))

    def save(self, path: str, pretty: Optional[bool] = True) -> None:
        """
        """

        folder_path = os.path.dirname(path)
        token_config_path = os.path.join(folder_path, "token_config.json")
        if folder_path and not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        self.tokenizer.save(path, pretty=pretty)
        self.token_config.save(token_config_path)
