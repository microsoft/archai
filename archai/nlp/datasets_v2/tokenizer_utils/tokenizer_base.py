# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tokenizer base class.
"""

import os
import json
from typing import List, Optional, Union

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
                 encode_special_tokens: Optional[bool] = True,
                 decode_special_tokens: Optional[bool] = True) -> None:
        """Initializes a base tokenizer class by setting attributes.

        Args:
            tokenizer: Instance of a Tokenizer from huggingface/tokenizers.
            trainer: Instance of a Trainer from huggingface/tokenizers.
            tokenizer_path: Path to the output pre-trained tokenizer file.
            token_config_path: Path to the output token's configuration file.
            min_freq: Minimum frequency of tokens (`0` for disabling argument).
            vocab_size: Maximum size of vocabulary.
            bos_token: Begin-of-sentence token.
            eos_token: End-of-sentence token.
            unk_token: Unknown token.
            sep_token: Separator token (used for separating two sequences).
            pad_token: Padding token.
            cls_token: Input class token.
            mask_token: Masked token.
            model_max_length: Maximum length of sequences.
            add_prefix_space: Whether a space should be added as a sequence prefix.
            add_prefix_new_line: Whether a new line should be added as a sequence prefix.
            lower_case: Applies lower case to all sequences.
            encode_special_tokens: Whether special tokens should be used to encode sequences.
            decode_special_tokens: Whether special tokens should be used to decode sequences.

        """

        # Tokenizer-based attributes
        self.tokenizer = tokenizer
        self.pre_trained_tokenizer = None
        self.trainer = trainer
        self.tokenizer_path = tokenizer_path
        self.token_config_path = token_config_path
        self.min_freq = min_freq
        self.vocab_size = vocab_size
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
                self.config = TokenConfig(**json.load(f))
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

    def __len__(self):
        return len(self.pre_trained_tokenizer)

    def encode(self, text: str) -> List[int]:
        return self.pre_trained_tokenizer.encode(self.config.pre_process(text),
                                                 add_special_tokens=self.encode_special_tokens)

    def decode(self, ids: Union[int, List[int]]) -> str:
        return self.pre_trained_tokenizer.decode(ids,
                                                 skip_special_tokens=self.decode_special_tokens)

    def ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        return self.pre_trained_tokenizer.convert_ids_to_tokens(ids,
                                                                skip_special_tokens=self.decode_special_tokens)

    def tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        return self.pre_trained_tokenizer.convert_tokens_to_ids(tokens,
                                                                skip_special_tokens=self.decode_special_tokens)
