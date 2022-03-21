# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Corpus base class, which serves for training/loading tokenizers and datasets.
"""

import os

from typing import Any, Union, Dict, List, Optional
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

from transformers import PreTrainedTokenizerFast
from archai.nlp.datasets_v2.dataset_loader import load_file_dataset, load_hub_dataset
from archai.nlp.datasets_v2.tokenizer_utils.token_config import TokenConfig
from archai.nlp.datasets_v2.tokenizer_utils.tokenizer_base import ArchaiTokenizer
from archai.nlp.datasets_v2.tokenizer_utils.word_tokenizer import WordTokenizer


class ArchaiCorpus:
    """ArchaiCorpus defines a single instance that is capable of training/loading a tokenizer,
        as well as producing a language modelling-ready dataset.

    """

    def __init__(self,
                 data_dir: str,
                 cache_dir: str,
                 data_config_name: Optional[str] = None,
                 data_type: Optional[str] = 'file',
                 data_files: Optional[Union[Dict[str, Any], List[str]]] = None,
                 data_split: Optional[List[str]] = None,
                 data_revision: Optional[str] = None,
                 data_features: Optional[List[str]] = None,
                 data_input_column_name: Optional[str] = 'text',
                 data_output_column_name: Optional[List[str]] = ['input_ids'],
                 data_random_seed: Optional[int] = 42,
                 data_n_samples: Optional[int] = 0,      
                 data_truncate: Optional[bool] = False,
                 data_padding: Optional[str] = 'do_not_pad',
                 from_stream: Optional[bool] = False,
                 refresh_cache: Optional[bool] = False,
                 vocab_type: Optional[str] = 'word',
                 vocab_min_freq: Optional[int] = 0,
                 vocab_size: Optional[int] = 10000,
                 tokenizer_max_length: Optional[int] = None) -> None:
        # Dataset-related attributes
        self.data_dir = data_dir
        self.data_config_name = data_config_name
        self.data_type = data_type
        self.data_files = data_files
        self.data_split = data_split
        self.data_revision = data_revision
        self.data_features = data_features
        self.data_input_column_name = data_input_column_name
        self.data_output_column_name = data_output_column_name
        self.data_random_seed = data_random_seed
        self.data_n_samples = data_n_samples
        self.data_truncate = data_truncate
        self.data_padding = data_padding

        # Cache/streaming-related attributes
        self.cache_dir = os.path.join(cache_dir, vocab_type, str(vocab_size))
        self.from_stream = from_stream
        self.refresh_cache = refresh_cache

        # Tokenizer and vocabulary-related attributes
        self.vocab_type = vocab_type
        self.vocab_min_freq = vocab_min_freq
        self.vocab_size = vocab_size
        self.tokenizer_max_length = tokenizer_max_length
        self.tokenizer_path = os.path.join(self.cache_dir, 'tokenizer.json')
        self.token_config_path = os.path.join(self.cache_dir, 'token_config.json')

    @property
    def is_tokenizer_trained(self) -> bool:
        return os.path.exists(self.tokenizer_path) and os.path.exists(self.token_config_path)

    def _load_dataset(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        # Loads dataset from user inputted files
        if self.data_type == 'file':
            return load_file_dataset(self.data_dir,
                                     self.cache_dir,
                                     data_files=self.data_files,
                                     split=self.data_split,
                                     features=self.data_features,
                                     from_stream=self.from_stream,
                                     refresh_cache=self.refresh_cache)

        # Loads dataset from Huggingface's datasets hub
        if self.data_type == 'hub':
            return load_hub_dataset(self.data_dir,
                                    self.data_config_name,
                                    self.cache_dir,
                                    split=self.data_split,
                                    revision=self.data_revision,
                                    from_stream=self.from_stream,
                                    refresh_cache=self.refresh_cache)

        raise NotImplementedError(f'data_type: {self.data_type} not supported yet.')

    def _create_tokenizer(self) -> ArchaiTokenizer:
        # Creates a word-based tokenizer
        if self.vocab_type == 'word':
            return WordTokenizer(tokenizer_path=self.tokenizer_path,
                                 token_config_path=self.token_config_path,
                                 min_freq=self.vocab_min_freq,
                                 vocab_size=self.vocab_size,
                                 model_max_length=self.tokenizer_max_length)
        
        raise NotImplementedError(f'vocab_type: {self.vocab_type} not supported yet.')

    def _encode(self,
                tokenizer: PreTrainedTokenizerFast,
                token_config: TokenConfig,
                dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]
               ) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        def _shuffle_and_select(dataset):
            if self.data_random_seed > -1:
                dataset = dataset.shuffle(self.data_random_seed)
            if self.data_n_samples > 0:
                dataset = dataset.select(range(self.data_n_samples))

            return dataset

        def _apply_tokenization(x):
            return tokenizer(token_config.pre_process(x[self.data_input_column_name]),
                             truncation=self.data_truncate,
                             padding=self.data_padding)

        if not hasattr(dataset, 'info'):
            for split, data in dataset.items():
                dataset[split] = _shuffle_and_select(data)
        else:
            dataset = _shuffle_and_select(dataset)

        # Encodes the dataset by applying the pre-processing and tokenization
        encoded_dataset = dataset.map(lambda x: _apply_tokenization(x), batched=True)
        encoded_dataset.set_format(type='torch', columns=self.data_output_column_name)

        return encoded_dataset

    def load(self) -> None:
        # Loads both dataset and tokenizer
        dataset = self._load_dataset()
        tokenizer = self._create_tokenizer()

        # If tokenizer has not been trained or if cache should be refreshed
        if not self.is_tokenizer_trained or self.refresh_cache:
            tokenizer.train(dataset, column_name=self.data_input_column_name)

        # Loads the tokenizer and encodes the dataset
        # Note that we pass the pre-trained tokenizer attribute to avoid hashing issues
        tokenizer.load()
        encoded_dataset = self._encode(tokenizer.pre_trained_tokenizer,
                                       tokenizer.config,
                                       dataset)

        # Attaches them as attributes
        self.dataset = encoded_dataset
        self.tokenizer = tokenizer
        self.token_config = tokenizer.config
            

def get_corpus(data_dir: str,
               cache_dir: str,
               data_config_name: Optional[str] = None,
               data_type: Optional[str] = 'file',
               data_files: Optional[Union[Dict[str, Any], List[str]]] = None,
               data_split: Optional[List[str]] = None,
               data_revision: Optional[str] = None,
               data_features: Optional[List[str]] = None,
               data_input_column_name: Optional[str] = 'text',
               data_output_column_name: Optional[List[str]] = ['input_ids'],
               data_random_seed: Optional[int] = 42,
               data_n_samples: Optional[int] = 0,      
               data_truncate: Optional[bool] = False,
               data_padding: Optional[str] = 'do_not_pad',
               from_stream: Optional[bool] = False,
               refresh_cache: Optional[bool] = False,
               vocab_type: Optional[str] = 'word',
               vocab_min_freq: Optional[int] = 0,
               vocab_size: Optional[int] = 10000,
               tokenizer_max_length: Optional[int] = None) -> ArchaiCorpus:
    corpus = ArchaiCorpus(data_dir,
                          cache_dir,
                          data_config_name=data_config_name,
                          data_type=data_type,
                          data_files=data_files,
                          data_split=data_split,
                          data_revision=data_revision,
                          data_features=data_features,
                          data_input_column_name=data_input_column_name,
                          data_output_column_name=data_output_column_name,
                          data_random_seed=data_random_seed,
                          data_n_samples=data_n_samples,
                          data_truncate=data_truncate,
                          data_padding=data_padding,
                          from_stream=from_stream,
                          refresh_cache=refresh_cache,
                          vocab_type=vocab_type,
                          vocab_min_freq=vocab_min_freq,
                          vocab_size=vocab_size,
                          tokenizer_max_length=tokenizer_max_length)
    corpus.load()

    return corpus
