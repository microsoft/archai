# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tokenizer base class.
"""

import os
import json
from typing import Callable, List, Optional

from datasets.arrow_dataset import Dataset
from tokenizers import Tokenizer
from tokenizers.trainers import Trainer
from transformers import PreTrainedTokenizerFast

from archai.nlp.datasets_v2.tokenizer_utils.token_config import TokenConfig

# Disables `tokenizers` parallelism due to process being forked
os.environ['TOKENIZERS_PARALLELISM'] = 'False'


class ArchaiTokenizer:
    """Base tokenizer class, used to define some common attributes
        and shared methods for training, saving and loading vocabularies.

    """

    def __init__(self,
                 tokenizer: Tokenizer,
                 trainer: Trainer,
                 tokenizer_path: Optional[str] = None,
                 token_config_path: Optional[str] = None,
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
        # Tokenizer-based attributes
        self.tokenizer = tokenizer
        self.pre_trained_tokenizer = None
        self.trainer = trainer
        self.tokenizer_path = tokenizer_path
        self.token_config_path = token_config_path
        self.min_freq = min_freq
        self.vocab_size = vocab_size
        self.delimiter = delimiter
        self.encode_special_tokens = encode_special_tokens
        self.decode_special_tokens = decode_special_tokens

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

    def train(self,
              dataset: Dataset,
              batch_size: Optional[int] = 10000,
              column_name: Optional[str] = 'text') -> None:
        def _batch_iterator(dataset, batch_size, column_name):
            for i in range(0, len(dataset), batch_size):
                yield dataset[i: i + batch_size][column_name]
        
        # Creates a training dataset iterator
        train_dataset = dataset['train']
        batch_dataset = _batch_iterator(train_dataset, batch_size, column_name)

        # Trains and saves a new tokenizer
        self.tokenizer.train_from_iterator(batch_dataset, self.trainer, len(train_dataset))
        self.tokenizer.save(self.tokenizer_path)

        # Also saves the token's config because it will be missed when loading
        # a pre-trained tokenizer
        with open(self.token_config_path, 'w') as f:
            json.dump(self.config.__dict__, f)

    def load(self) -> None:
        # Attempts to load the token's configuration because it will be missed
        # when creating the PreTrainedTokenizerFast from file
        try:
            with open(self.token_config_path, 'r') as f:
                config = json.load(f)
                self.config = TokenConfig(**config)
        except:
            raise FileNotFoundError(f'{self.token_config_path} could not be found.')
        
        # Attempts to load the pre-trained tokenizer (compatible with `transformers`)
        try:
            self.pre_trained_tokenizer = PreTrainedTokenizerFast(model_max_length=self.config.model_max_length,
                                                                 bos_token=self.config.bos_token,
                                                                 eos_token=self.config.eos_token,
                                                                 unk_token=self.config.unk_token,
                                                                 sep_token=self.config.sep_token,
                                                                 pad_token=self.config.pad_token,
                                                                 cls_token=self.config.cls_token,
                                                                 mask_token=self.config.mask_token,
                                                                 tokenizer_file=self.tokenizer_path)
        except:
            raise FileNotFoundError(f'{self.tokenizer_path} could not be found.')

    # def encode(self, text: str) -> List[int]:
    #     return self.tokenizer.encode(self.config.pre_process(text),
    #                                  add_special_tokens=self.encode_special_tokens)

    # def decode(self, ids: List[int]) -> str:
    #     return self.tokenizer.decode(ids, skip_special_tokens=self.decode_special_tokens)

    # def id_to_token(self, id: int) -> str:
    #     return self.tokenizer.id_to_token(id)

    # def ids_to_tokens(self, ids: List[int]) -> List[str]:
    #     return [self.id_to_token(id) for id in ids]

    # def token_to_id(self, token: str) -> int:
    #     return self.tokenizer.token_to_id(token)

    # def tokens_to_ids(self, tokens: List[str]) -> List[int]:
    #     return [self.token_to_id(token) for token in tokens]
    