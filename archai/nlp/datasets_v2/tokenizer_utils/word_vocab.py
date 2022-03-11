# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

from archai.nlp.datasets_v2.tokenizer_utils.vocab_base import Vocab

from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Punctuation, Sequence, Whitespace
from tokenizers.trainers import WordLevelTrainer


class WordVocab(Vocab):
    """
    """

    def __init__(self, vocab_path: str) -> None:
        """
        """

        model = WordLevel(unk_token='UNK')
        trainer = WordLevelTrainer(special_tokens=['UNK'])

        super().__init__(model, trainer, vocab_path)
