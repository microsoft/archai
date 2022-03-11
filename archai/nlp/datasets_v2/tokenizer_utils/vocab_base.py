# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

import os
from typing import Optional

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer

from archai.nlp.datasets_v2.tokenizer_utils.token_config import TokenConfig


class Vocab:
    """
    """

    def __init__(self,
                 model,
                 trainer,
                 vocab_path,
                 min_freq: Optional[int] = 0,
                 vocab_size: Optional[int] = 10000,
                 bos_token: Optional[str] = None,
                 eos_token: Optional[str] = None,
                 unk_token: Optional[str] = None,
                 pad_token: Optional[str] = None,
                 add_prefix_space: Optional[bool] = False,
                 add_prefix_new_line: Optional[bool] = False,
                 lower_case: Optional[bool] = False,
                 delimiter: Optional[str] = None,
                 encode_special_tokens: Optional[bool] = True,
                 decode_special_tokens: Optional[bool] = True):
        """
        """

        #
        self.config = TokenConfig(bos_token=bos_token,
                                  eos_token=eos_token,
                                  unk_token=unk_token,
                                  pad_token=pad_token,
                                  add_prefix_space=add_prefix_space,
                                  add_prefix_new_line=add_prefix_new_line,
                                  lower_case=lower_case)

        #
        self.model = model
        self.trainer = trainer
        self.vocab_path = vocab_path

        #
        self.min_freq = min_freq
        self.vocab_size = vocab_size
        self.delimiter = delimiter
        self.encode_special_tokens = encode_special_tokens
        self.decode_special_tokens = decode_special_tokens

    def train(self, dataset):
        """
        """

        def _batch_iterator(dataset, batch_size=10000, column_name='text'):
            for i in range(0, len(dataset), batch_size):
                yield dataset[i: i + batch_size][column_name]

        #
        vocab = Tokenizer(self.model)

        #
        vocab.train_from_iterator(_batch_iterator(dataset['train']),
                                      self.trainer,
                                      len(dataset))
        vocab.save(self.vocab_path)

    def is_trained(self):
        """
        """
        
        return os.path.exists(self.vocab_path)

    def load(self):
        """
        """

        if self.is_trained():
            vocab = PreTrainedTokenizerFast(tokenizer_file=self.vocab_path)
            vocab.unk_token = 'UNK'
            return vocab
        else:
            raise FileNotFoundError()
