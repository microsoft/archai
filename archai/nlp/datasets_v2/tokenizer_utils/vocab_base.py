# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Vocabulary (tokenizer) base class.
"""

import os
import json
from typing import Optional

from datasets.arrow_dataset import Dataset
from tokenizers import Tokenizer
from tokenizers.trainers import Trainer
from transformers import PreTrainedTokenizerFast

from archai.nlp.datasets_v2.tokenizer_utils.token_config import TokenConfig

# Disables `tokenizers` parallelism due to process being forked
os.environ['TOKENIZERS_PARALLELISM'] = 'False'


class Vocab:
    """Base vocabulary (tokenizer) class, used to define some common attributes
        and shared methods for training, saving and loading vocabularies.

    """

    def __init__(self,
                 min_freq: Optional[int] = 0,
                 vocab_size: Optional[int] = 10000,
                 bos_token: Optional[str] = None,
                 eos_token: Optional[str] = None,
                 unk_token: Optional[str] = None,
                 sep_token: Optional[str] = None,
                 pad_token: Optional[str] = None,
                 cls_token: Optional[str] = None,
                 mask_token: Optional[str] = None,
                 model_max_length: Optional[int] = None,
                 add_prefix_space: Optional[bool] = False,
                 add_prefix_new_line: Optional[bool] = False,
                 lower_case: Optional[bool] = False,
                 delimiter: Optional[str] = None,
                 encode_special_tokens: Optional[bool] = True,
                 decode_special_tokens: Optional[bool] = True) -> None:
        # Token-related configuration
        self.config = TokenConfig(bos_token=bos_token,
                                  eos_token=eos_token,
                                  unk_token=unk_token,
                                  sep_token=sep_token,
                                  pad_token=pad_token,
                                  cls_token=cls_token,
                                  mask_token=mask_token,
                                  model_max_length=model_max_length,
                                  add_prefix_space=add_prefix_space,
                                  add_prefix_new_line=add_prefix_new_line,
                                  lower_case=lower_case)

        # Vocabulary-related attributes
        self.min_freq = min_freq
        self.vocab_size = vocab_size
        self.delimiter = delimiter
        self.encode_special_tokens = encode_special_tokens
        self.decode_special_tokens = decode_special_tokens

    def train(self,
              output_path: str,
              dataset: Dataset,
              batch_size: Optional[int] = 10000,
              column_name: Optional[str] = 'text') -> None:
        assert isinstance(self.vocab, Tokenizer), '`vocab` must be derived from the Tokenizer class.'
        assert isinstance(self.trainer, Trainer), '`trainer` must be derived from the Trainer class.'

        def _batch_iterator(dataset, batch_size, column_name):
            for i in range(0, len(dataset), batch_size):
                yield dataset[i: i + batch_size][column_name]
        
        # Creates a training dataset iterator
        train_dataset = dataset['train']
        batch_dataset = _batch_iterator(train_dataset, batch_size, column_name)

        # Trains and saves a new vocabulary
        self.vocab.train_from_iterator(batch_dataset, self.trainer, len(train_dataset))
        self.vocab.save(output_path)

        # Also saves the token's config because it will be missed when loading
        # a pre-trained vocabulary
        config_output_path = os.path.join(os.path.dirname(output_path), 'token_config.json')
        with open(config_output_path, 'w') as f:
            json.dump(self.config.__dict__, f)
