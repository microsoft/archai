# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""GPT-2-based tokenizer."""

from typing import Optional

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer

from archai.nlp.datasets.hf.tokenizer_utils.token_config import (
    SPECIAL_TOKENS,
    TokenConfig,
)
from archai.nlp.datasets.hf.tokenizer_utils.tokenizer_base import TokenizerBase


class GPT2Tokenizer(TokenizerBase):
    """A GPT-2-based tokenizer for processing text data."""

    def __init__(self, vocab_size: Optional[int] = 50257, min_frequency: Optional[int] = 0) -> None:
        """Define the tokenization pipeline.

        Args:
            vocab_size: The maximum size of the vocabulary to be generated.
            min_frequency: The minimum frequency of tokens to be included in the vocabulary.

        """

        token_config = TokenConfig(
            bos_token=SPECIAL_TOKENS["bos_token"],
            eos_token=SPECIAL_TOKENS["gpt2_eos_token"],
            unk_token=SPECIAL_TOKENS["unk_token"],
            pad_token=SPECIAL_TOKENS["pad_token"],
        )

        tokenizer = Tokenizer(BPE(continuing_subword_prefix="", end_of_word_suffix=""))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=token_config.special_tokens,
        )

        super().__init__(token_config, tokenizer, trainer)
