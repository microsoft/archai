# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Utilities for tokenization pipelines with huggingface/tokenizers.
"""

from __future__ import annotations

import os
from typing import Iterator, List, Optional

from tokenizers import Tokenizer
from tokenizers.trainers import Trainer

from archai.nlp.datasets.hf_datasets.tokenizer_utils.token_config import TokenConfig


class TokenizerBase:
    """Implements a base class for a customizable tokenization pipeline."""

    def __init__(self, token_config: TokenConfig, tokenizer: Tokenizer, trainer: Trainer) -> None:
        """Attaches attributes to class.

        Args:
            token_config: TokenConfig class with tokens' configuration.
            tokenizer: Tokenizer class with model from huggingface/tokenizers.
            trainer: Trainer class from huggingface/tokenizers.

        """

        self.token_config = token_config
        self.tokenizer = tokenizer
        self.trainer = trainer

    def train(self, files: List[str]) -> None:
        """Trains tokenizer from a list of files.

        Args:
            files: List of paths to input files.

        """

        return self.tokenizer.train(files, trainer=self.trainer)

    def train_from_iterator(self, iterator: Iterator) -> None:
        """Trains tokenizer from in-memory data.

        Args:
            iterator: Raw data to be tokenized.

        """

        return self.tokenizer.train_from_iterator(iterator, trainer=self.trainer, length=len(iterator))

    def save(self, path: str, pretty: Optional[bool] = True) -> None:
        """Saves the pre-trained tokenizer.

        Args:
            path: Path to tokenizer file.
            pretty: Whether the JSON file should be pretty formatted.

        """

        folder_path = os.path.dirname(path)
        token_config_path = os.path.join(folder_path, "token_config.json")
        if folder_path and not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.tokenizer.save(path, pretty=pretty)
        self.token_config.save(token_config_path)
