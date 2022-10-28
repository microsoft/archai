# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""TransformerXL-based tokenizer.
"""

from typing import Optional

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Punctuation, Sequence, Whitespace
from tokenizers.trainers import WordLevelTrainer

from archai.nlp.datasets.hf.tokenizer_utils.token_config import (
    SPECIAL_TOKENS,
    TokenConfig,
)
from archai.nlp.datasets.hf.tokenizer_utils.tokenizer_base import TokenizerBase


class TransfoXLTokenizer(TokenizerBase):
    """Implements a Transformer-XL-based tokenizer."""

    def __init__(self, vocab_size: Optional[int] = 267735, min_frequency: Optional[int] = 0) -> None:
        """Defines the tokenization pipeline.

        Args:
            vocab_size: Maximum size of vocabulary.
            min_frequency: Minimum frequency of tokens.

        """

        token_config = TokenConfig(
            eos_token=SPECIAL_TOKENS["eos_token"],
            unk_token=SPECIAL_TOKENS["unk_token"],
            sep_token=SPECIAL_TOKENS["transfo_xl_sep_token"],
            pad_token=SPECIAL_TOKENS["pad_token"],
        )

        tokenizer = Tokenizer(WordLevel(unk_token=token_config.unk_token))
        tokenizer.pre_tokenizer = Sequence([Punctuation(), Whitespace()])

        trainer = WordLevelTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=token_config.special_tokens,
        )

        super().__init__(token_config, tokenizer, trainer)
