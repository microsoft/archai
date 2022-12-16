# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""CodeGen-based tokenizer."""

from typing import Optional

from tokenizers import AddedToken, Tokenizer
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


class CodeGenTokenizer(TokenizerBase):
    """A CodeGen-based tokenizer for processing text data (including code)."""

    def __init__(self, vocab_size: Optional[int] = 50257, min_frequency: Optional[int] = 0) -> None:
        """Define the tokenization pipeline.

        Args:
            vocab_size: The maximum size of the vocabulary to be generated.
            min_frequency: The minimum frequency of tokens to be included in the vocabulary.

        """

        token_config = TokenConfig(
            bos_token=SPECIAL_TOKENS["gpt2_eos_token"],
            eos_token=SPECIAL_TOKENS["gpt2_eos_token"],
            unk_token=SPECIAL_TOKENS["gpt2_eos_token"],
            pad_token=SPECIAL_TOKENS["pad_token"],
        )

        tokenizer = Tokenizer(BPE(continuing_subword_prefix="", end_of_word_suffix=""))
        tokenizer.add_special_tokens(token_config.special_tokens)
        tokenizer.add_tokens([AddedToken(" " * i) for i in range(2, 32)] + [AddedToken("\t" * i) for i in range(2, 10)])
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            continuing_subword_prefix="",
            end_of_word_suffix="",
            special_tokens=token_config.special_tokens,
        )

        super().__init__(token_config, tokenizer, trainer)
