# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Utilities for tokenization pipelines with huggingface/tokenizers."""

from __future__ import annotations

import os
from typing import Iterator, List, Optional

from overrides.enforce import EnforceOverrides
from tokenizers import Tokenizer
from tokenizers.trainers import Trainer

from archai.nlp.datasets.hf.tokenizer_utils.token_config import TokenConfig


class TokenizerBase(EnforceOverrides):
    """A base class for building a customizable tokenization pipeline
    using the huggingface/tokenizers library.

    """

    def __init__(self, token_config: TokenConfig, tokenizer: Tokenizer, trainer: Trainer) -> None:
        """Initializes the `TokenizerBase` object by attaching attributes.

        Args:
            token_config: A `TokenConfig` object with tokens' configuration.
            tokenizer: A `Tokenizer` object from the huggingface/transformers library.
            trainer: A `Trainer` object from the huggingface/transformers library.

        """

        assert isinstance(token_config, TokenConfig), "`token_config` must be an instance of `TokenConfig`."
        assert isinstance(tokenizer, Tokenizer), "`tokenizer` must be an instance of `Tokenizer`."
        assert isinstance(trainer, Trainer), "`trainer` must be an instance of `Trainer`."

        self.token_config = token_config
        self.tokenizer = tokenizer
        self.trainer = trainer

    def train(self, files: List[str]) -> None:
        """Train the tokenizer on a list of text files.

        Args:
            files: A list of paths to input files.

        """

        return self.tokenizer.train(files, trainer=self.trainer)

    def train_from_iterator(
        self, iterator: Iterator, batch_size: Optional[int] = 10000, column_name: Optional[str] = "text"
    ) -> None:
        """Train the tokenizer on in-memory data.

        Args:
            iterator: Raw data to be tokenized.
            batch_size: Size of each batch.
            column_name: Name of column that should be retrieved.

        """

        def _batch_iterator(iterator: Iterator) -> Iterator:
            for i in range(0, len(iterator), batch_size):
                yield iterator[i : i + batch_size][column_name]

        return self.tokenizer.train_from_iterator(_batch_iterator(iterator), trainer=self.trainer, length=len(iterator))

    def save(self, path: str, pretty: Optional[bool] = True) -> None:
        """Save the pre-trained tokenizer to a file.

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
