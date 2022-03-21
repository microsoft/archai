# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Corpus base class, which serves for training/loading datasets and tokenizers.
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
                 data_random_seed: Optional[int] = 0,
                 data_n_samples: Optional[int] = 0,      
                 data_truncate: Optional[bool] = True,
                 data_padding: Optional[str] = 'max_length',
                 from_stream: Optional[bool] = False,
                 data_refresh_cache: Optional[bool] = False,
                 vocab_type: Optional[str] = 'word',
                 vocab_min_freq: Optional[int] = 0,
                 vocab_size: Optional[int] = 10000,
                 tokenizer_max_length: Optional[int] = 192,
                 tokenizer_refresh_cache: Optional[bool] = False) -> None:
        """Initializes the corpus by setting attributes.

        Args:
            data_dir: Directory of the dataset to be loaded.
            cache_dir: Directory of where the cache should be stored.
            data_config_name: Name of dataset's configuration to be loaded.
            data_type: Type of dataset to be loaded (`file` or `hub`).
            data_files: Files that should be loaded from `data_dir`.
            data_split: Specific splits that should be loaded (`train`, `val` or `test`).
            data_revision: Version of the dataset to be loaded.
            data_features: Custom features (column names) that should be loaded.
            data_input_column_name: Input column name of dataset (usually `text`).
            data_output_column_name: Output column name of dataset (usually `input_ids`).
            data_random_seed: Random seed for shuffling the dataset (`0` for disabling argument).
            data_n_samples: Number of samples to subsample the dataset (`0` for disabling argument).
            data_truncate: Whether exceed `tokenizer_max_length` sequences should be truncated.
            data_padding: Whether non-exceed `tokenizer_max_length` should be padded (`do_not_pad`, `max_length` or `longest`).
            data_refresh_cache: Whether dataset cache should be refreshed or not.
            from_stream: Whether dataset should be streamed or not.
            vocab_type: Type of vocabulary (`word`, `bbpe` or `gpt2`).
            vocab_min_freq: Minimum frequency of tokens (`0` for disabling argument).
            vocab_size: Maximum size of vocabulary.
            tokenizer_max_length: Maximum length of tokenization sequences (`None` for disabling argument).
            tokenizer_refresh_cache: Whether tokenizer cache should be refreshed or not.

        """

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
        self.data_refresh_cache = data_refresh_cache
        self.tokenizer_refresh_cache = tokenizer_refresh_cache

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
        # Loads dataset from inputted files
        if self.data_type == 'file':
            return load_file_dataset(self.data_dir,
                                     self.cache_dir,
                                     data_files=self.data_files,
                                     split=self.data_split,
                                     features=self.data_features,
                                     from_stream=self.from_stream,
                                     refresh_cache=self.data_refresh_cache)

        # Loads dataset from Huggingface's datasets hub
        if self.data_type == 'hub':
            return load_hub_dataset(self.data_dir,
                                    self.data_config_name,
                                    self.cache_dir,
                                    split=self.data_split,
                                    revision=self.data_revision,
                                    from_stream=self.from_stream,
                                    refresh_cache=self.data_refresh_cache)

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
        def _shuffle_and_select(d):
            if self.data_random_seed > 0:
                d = d.shuffle(self.data_random_seed)
            if self.data_n_samples > 0:
                d = d.select(range(self.data_n_samples))

            return d

        def _apply_tokenization(x):
            return tokenizer(token_config.pre_process(x[self.data_input_column_name]),
                             truncation=self.data_truncate,
                             padding=self.data_padding)

        # Checks if dataset is a DatasetDict (usually contains multiple splits)
        if isinstance(dataset, DatasetDict):
            for split, data in dataset.items():
                dataset[split] = _shuffle_and_select(data)
        else:
            dataset = _shuffle_and_select(dataset)

        # Encodes the dataset by applying pre-processing and tokenization
        encoded_dataset = dataset.map(lambda x: _apply_tokenization(x), batched=True)
        encoded_dataset.set_format(type='torch', columns=self.data_output_column_name)
        encoded_dataset.save_to_disk(self.cache_dir)

        return encoded_dataset

    def load(self) -> None:
        dataset = self._load_dataset()
        tokenizer = self._create_tokenizer()

        # Tokenizer should be re-trained it it has not been trained yet
        # or if the cache should be refreshed
        if not self.is_tokenizer_trained or self.tokenizer_refresh_cache:
            tokenizer.train(dataset, column_name=self.data_input_column_name)

        # Loads the tokenizer and encodes the dataset
        # Note that we pass the pre-trained tokenizer attribute to avoid hashing issues
        tokenizer.load()
        encoded_dataset = self._encode(tokenizer.pre_trained_tokenizer,
                                       tokenizer.config,
                                       dataset)

        # Do not forget to attach them as attributes
        self.dataset = encoded_dataset
        self.tokenizer = tokenizer
        self.token_config = tokenizer.config
