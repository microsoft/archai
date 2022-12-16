# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Character-based tokenizer."""

from typing import Optional

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer

from archai.nlp.datasets.hf.tokenizer_utils.token_config import (
    SPECIAL_TOKENS,
    TokenConfig,
)
from archai.nlp.datasets.hf.tokenizer_utils.tokenizer_base import TokenizerBase


class CharTokenizer(TokenizerBase):
    """A character-based tokenizer for processing text data."""

    def __init__(self, vocab_size: Optional[int] = 256, min_frequency: Optional[int] = 0) -> None:
        """Define the tokenization pipeline.

        Args:
            vocab_size: The maximum size of the vocabulary to be generated.
            min_frequency: The minimum frequency of tokens to be included in the vocabulary.

        """

        token_config = TokenConfig(
            bos_token=SPECIAL_TOKENS["bos_token"],
            eos_token=SPECIAL_TOKENS["eos_token"],
            unk_token=SPECIAL_TOKENS["unk_token"],
            sep_token=SPECIAL_TOKENS["sep_token"],
            pad_token=SPECIAL_TOKENS["pad_token"],
        )

        tokenizer = Tokenizer(WordLevel(unk_token=token_config.unk_token))
        tokenizer.pre_tokenizer = Split("", "isolated")
        tokenizer.post_processor = TemplateProcessing(
            single=f"{token_config.bos_token} $A {token_config.eos_token}",
            pair=f"{token_config.bos_token} $A {token_config.sep_token} $B:1 {token_config.eos_token}:1",
            special_tokens=[
                (token_config.bos_token, 0),
                (token_config.eos_token, 1),
                (token_config.sep_token, 3),
            ],
        )
        tokenizer.enable_padding(pad_id=4, pad_token=token_config.pad_token)

        trainer = WordLevelTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=token_config.special_tokens,
        )

        super().__init__(token_config, tokenizer, trainer)
