# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Character-based tokenizer.
"""

from typing import Optional

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer

from archai.nlp.tokenizer import SPECIAL_TOKENS, ArchaiTokenConfig, ArchaiTokenizer


class CharTokenizer(ArchaiTokenizer):
    """Creates a customizable character-based tokenization pipeline."""

    def __init__(self, vocab_size: Optional[int] = 256, min_frequency: Optional[int] = 0) -> None:
        """Defines the tokenization pipeline.

        Args:
            vocab_size: Maximum size of vocabulary.
            min_frequency: Minimum frequency of tokens.

        """

        # Initializes token's configuration, tokenizer and trainer
        token_config = ArchaiTokenConfig(
            bos_token=SPECIAL_TOKENS["bos_token"],
            eos_token=SPECIAL_TOKENS["eos_token"],
            unk_token=SPECIAL_TOKENS["unk_token"],
            sep_token=SPECIAL_TOKENS["sep_token"],
            pad_token=SPECIAL_TOKENS["pad_token"],
        )
        tokenizer = Tokenizer(WordLevel(unk_token=token_config.unk_token))
        trainer = WordLevelTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=token_config.special_tokens,
        )

        super().__init__(token_config, tokenizer, trainer)

        # Pre- and post-processing templates
        self.tokenizer.pre_tokenizer = Split("", "isolated")
        self.tokenizer.post_processor = TemplateProcessing(
            single=f"{token_config.bos_token} $A {token_config.eos_token}",
            pair=f"{token_config.bos_token} $A {token_config.sep_token} $B:1 {token_config.eos_token}:1",
            special_tokens=[
                (token_config.bos_token, 0),
                (token_config.eos_token, 1),
                (token_config.sep_token, 3),
            ],
        )

        # Enables padding
        self.tokenizer.enable_padding(pad_id=4, pad_token=token_config.pad_token)
