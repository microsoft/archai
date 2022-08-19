# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""BERT-based tokenizer.
"""

from typing import Optional

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, Lowercase, Sequence, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer

from archai.nlp.tokenizer import SPECIAL_TOKENS, ArchaiTokenConfig, ArchaiTokenizer


class BERTTokenizer(ArchaiTokenizer):
    """Creates a customizable BERT-based tokenization pipeline."""

    def __init__(self, vocab_size: Optional[int] = 30522, min_frequency: Optional[int] = 0) -> None:
        """Defines the tokenization pipeline.

        Args:
            vocab_size: Maximum size of vocabulary.
            min_frequency: Minimum frequency of tokens.

        """

        # Initializes token's configuration, tokenizer and trainer
        token_config = ArchaiTokenConfig(
            unk_token=SPECIAL_TOKENS["unk_token"],
            sep_token=SPECIAL_TOKENS["sep_token"],
            pad_token=SPECIAL_TOKENS["pad_token"],
            cls_token=SPECIAL_TOKENS["cls_token"],
            mask_token=SPECIAL_TOKENS["mask_token"],
        )
        tokenizer = Tokenizer(WordPiece(unk_token=token_config.unk_token))
        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=token_config.special_tokens,
        )

        super().__init__(token_config, tokenizer, trainer)

        # Normalizers, pre- and post-processing templates
        self.tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.post_processor = TemplateProcessing(
            single=f"{token_config.cls_token} $A {token_config.sep_token}",
            pair=f"{token_config.cls_token} $A {token_config.sep_token} $B:1 {token_config.sep_token}:1",
            special_tokens=[(token_config.sep_token, 1), (token_config.cls_token, 3)],
        )
