# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""GPT-2-based tokenizer.
"""

import os
from typing import Optional

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer

from archai.nlp.datasets.tokenizers.token_config import SPECIAL_TOKENS, TokenConfig


class GPT2Tokenizer(Tokenizer):
    """Creates a GPT-2-based tokenizer."""

    def __init__(
        self, vocab_size: Optional[int] = 50257, min_frequency: Optional[int] = 0
    ) -> None:
        """Defines the tokenization pipeline.

        Args:
            vocab_size: Maximum size of vocabulary.
            min_frequency: Minimum frequency of tokens.

        """

        super().__init__(BPE(continuing_subword_prefix="", end_of_word_suffix=""))

        self.token_config = TokenConfig(
            bos_token=SPECIAL_TOKENS["bos_token"],
            eos_token=SPECIAL_TOKENS["gpt2_eos_token"],
            unk_token=SPECIAL_TOKENS["unk_token"],
            pad_token=SPECIAL_TOKENS["pad_token"],
        )

        self.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.post_processor = ByteLevelProcessor(trim_offsets=False)
        self.decoder = ByteLevelDecoder()

        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=self.token_config.special_tokens,
        )

    def train_from_iterator(self, iterator) -> None:
        """
        """
                
        return super().train_from_iterator(iterator, self.trainer, len(iterator))

    def save(self, path: str, pretty: Optional[bool] = True) -> None:
        """Saves the pre-trained tokenizer and token's configuration to disk.

        Args:
            path: Path to where tokenizer should be saved.

        """

        folder_path = os.path.dirname(path)
        if folder_path and not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        token_config_path = os.path.join(folder_path, "token_config.json")
        self.token_config.save(token_config_path)

        return super().save(path, pretty=pretty)
