# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""GPT-2-based tokenizer.
"""

from typing import Optional

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer


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

        # Pre-, post-processing and decoder templates
        self.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.post_processor = ByteLevelProcessor(trim_offsets=False)
        self.decoder = ByteLevelDecoder()

        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            # special_tokens=token_config.special_tokens,
        )

        
        # token_config = ArchaiTokenConfig(
        #     bos_token=SPECIAL_TOKENS["bos_token"],
        #     eos_token=SPECIAL_TOKENS["gpt2_eos_token"],
        #     unk_token=SPECIAL_TOKENS["unk_token"],
        #     pad_token=SPECIAL_TOKENS["pad_token"],
        # )

    def train_from_iterator(self, iterator):
        """
        """

        def _batch_iterator(
            dataset,
            batch_size=10000,
            column_name="text",
        ):
            """Iterates over dataset to provide batches.

            Args:
                dataset: Dataset that should be iterated over.
                batch_size: Size of each batch.
                column_name: Name of column that should be retrieved.

            Yields:
                (Dataset): Batch of data based on size and `column_name`.

            """

            for i in range(0, len(dataset), batch_size):
                yield dataset[i : i + batch_size][column_name]
                
        return super().train_from_iterator(iterator, self.trainer, len(iterator))
