# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""BERT-based tokenizer."""

from typing import Optional

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, Lowercase, Sequence, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer

from archai.nlp.datasets.hf.tokenizer_utils.token_config import (
    SPECIAL_TOKENS,
    TokenConfig,
)
from archai.nlp.datasets.hf.tokenizer_utils.tokenizer_base import TokenizerBase


class BertTokenizer(TokenizerBase):
    """A BERT-based tokenizer for processing text data."""

    def __init__(self, vocab_size: Optional[int] = 30522, min_frequency: Optional[int] = 0) -> None:
        """Define the tokenization pipeline.

        Args:
            vocab_size: The maximum size of the vocabulary to be generated.
            min_frequency: The minimum frequency of tokens to be included in the vocabulary.

        """

        token_config = TokenConfig(
            unk_token=SPECIAL_TOKENS["unk_token"],
            sep_token=SPECIAL_TOKENS["sep_token"],
            pad_token=SPECIAL_TOKENS["pad_token"],
            cls_token=SPECIAL_TOKENS["cls_token"],
            mask_token=SPECIAL_TOKENS["mask_token"],
        )

        tokenizer = Tokenizer(WordPiece(unk_token=token_config.unk_token))
        tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = TemplateProcessing(
            single=f"{token_config.cls_token} $A {token_config.sep_token}",
            pair=f"{token_config.cls_token} $A {token_config.sep_token} $B:1 {token_config.sep_token}:1",
            special_tokens=[(token_config.sep_token, 1), (token_config.cls_token, 3)],
        )

        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=token_config.special_tokens,
        )

        super().__init__(token_config, tokenizer, trainer)
