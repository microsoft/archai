# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Utilities for loading pre-trained tokenizers."""

from typing import Optional

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from archai.nlp.datasets.hf.tokenizer_utils.token_config import TokenConfig


class ArchaiPreTrainedTokenizerFast(PreTrainedTokenizerFast):
    """A pre-trained fast tokenizer with the configuration of special tokens."""

    def __init__(
        self,
        *args,
        tokenizer_file: Optional[str] = None,
        token_config_file: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the `ArchaiPreTrainedTokenizerFast` class with
        additional keyword arguments.

        Args:
            tokenizer_file: Path to the tokenizer's file.
            token_config_file: Path to the token's configuration file.

        """

        token_config = TokenConfig.from_file(token_config_file)
        token_config_dict = token_config.to_dict()
        token_config_dict.update(kwargs)

        super().__init__(*args, tokenizer_file=tokenizer_file, **token_config_dict)
