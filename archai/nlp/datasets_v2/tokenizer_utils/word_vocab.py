# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Word-based vocabulary.
"""

from typing import Optional
from archai.nlp.datasets_v2.tokenizer_utils.vocab_base import Vocab

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Punctuation, Sequence, Whitespace
from tokenizers.trainers import WordLevelTrainer


class WordVocab(Vocab):
    """Word-based vocabulary, where sequences are split according to their
        punctuation and whitespaces.

    """

    def __init__(self,
                 min_freq: Optional[int] = 0,
                 vocab_size: Optional[int] = 10000,
                 bos_token: Optional[str] = None,
                 eos_token: Optional[str] = '<eos>',
                 unk_token: Optional[str] = '<unk>',
                 pad_token: Optional[str] = None,
                 mask_token: Optional[str] = None,
                 add_prefix_space: Optional[bool] = False,
                 add_prefix_new_line: Optional[bool] = False,
                 lower_case: Optional[bool] = False,
                 delimiter: Optional[str] = None,
                 encode_special_tokens: Optional[bool] = True,
                 decode_special_tokens: Optional[bool] = True) -> None:
        super().__init__(min_freq=min_freq,
                         vocab_size=vocab_size,
                         bos_token=bos_token,
                         eos_token=eos_token,
                         unk_token=unk_token,
                         pad_token=pad_token,
                         mask_token=mask_token,
                         add_prefix_space=add_prefix_space,
                         add_prefix_new_line=add_prefix_new_line,
                         lower_case=lower_case,
                         delimiter=delimiter,
                         encode_special_tokens=encode_special_tokens,
                         decode_special_tokens=decode_special_tokens)

        # Word-level vocabulary, pre-tokenizers and trainer
        self.vocab = Tokenizer(WordLevel())
        self.vocab.pre_tokenizer = Sequence([Punctuation(), Whitespace()])
        self.trainer = WordLevelTrainer(vocab_size=vocab_size,
                                        min_frequency=min_freq,
                                        special_tokens=self.config.special_tokens)
