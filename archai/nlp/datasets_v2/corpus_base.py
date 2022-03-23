# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Corpus base class, which serves for training/loading datasets and tokenizers.
"""

import os

from typing import Any, Union, Dict, List, Optional
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

from archai.nlp.datasets_v2.dataset_loader import load_file_dataset, load_hub_dataset
from archai.nlp.datasets_v2.tokenizer_utils.token_config import TokenConfig
from archai.nlp.datasets_v2.tokenizer_utils.tokenizer_base import ArchaiPreTrainedTokenizer, ArchaiTokenizer
from archai.nlp.datasets_v2.tokenizer_utils.bbpe_tokenizer import BBPETokenizer
from archai.nlp.datasets_v2.tokenizer_utils.gpt2_tokenizer import GPT2Tokenizer
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
                 data_contiguous: Optional[bool] = True,      
                 data_truncate: Optional[bool] = False,
                 data_padding: Optional[str] = 'do_not_pad',
                 data_from_stream: Optional[bool] = False,
                 data_refresh_cache: Optional[bool] = False,
                 vocab_type: Optional[str] = 'word',
                 vocab_min_freq: Optional[int] = 0,
                 vocab_size: Optional[int] = 10000,
                 tokenizer_max_length: Optional[int] = None,
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
            data_contiguous: Whether dataset should be ordered as a single and contiguous example.
            data_truncate: Whether exceed `tokenizer_max_length` sequences should be truncated.
            data_padding: Whether non-exceed `tokenizer_max_length` should be padded (`do_not_pad`, `max_length` or `longest`).
            data_refresh_cache: Whether dataset cache should be refreshed or not.
            data_from_stream: Whether dataset should be streamed or not.
            vocab_type: Type of vocabulary (`word`, `bbpe` or `gpt2`).
            vocab_min_freq: Minimum frequency of tokens (`0` for disabling argument).
            vocab_size: Maximum size of vocabulary.
            tokenizer_max_length: Maximum length of tokenization sequences (`None` for disabling argument).
            tokenizer_refresh_cache: Whether tokenizer cache should be refreshed or not.

        """

        assert not (data_contiguous and (data_truncate or data_padding != 'do_not_pad')), \
        'If data_contiguous is `True`, data_truncate should be `False` and data_padding should be `do_not_pad`'

        # Dataset-related attributes
        self.cache_dir = os.path.join(cache_dir, vocab_type, str(vocab_size))
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
        self.data_contiguous = data_contiguous
        self.data_truncate = data_truncate
        self.data_padding = data_padding
        self.data_refresh_cache = data_refresh_cache
        self.data_from_stream = data_from_stream

        # Vocabulary-related attributes
        self.vocab_type = vocab_type
        self.vocab_min_freq = vocab_min_freq
        self.vocab_size = vocab_size

        # Tokenizer-related attributes
        self.tokenizer_max_length = tokenizer_max_length
        self.tokenizer_path = os.path.join(self.cache_dir, 'tokenizer.json')
        self.token_config_path = os.path.join(self.cache_dir, 'token_config.json')
        self.tokenizer_refresh_cache = tokenizer_refresh_cache

        # Uninitialized attributes
        self.dataset = None
        self.tokenizer = None
        self.token_config = None

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
                                     from_stream=self.data_from_stream,
                                     refresh_cache=self.data_refresh_cache)

        # Loads dataset from Huggingface's datasets hub
        if self.data_type == 'hub':
            return load_hub_dataset(self.data_dir,
                                    self.data_config_name,
                                    self.cache_dir,
                                    split=self.data_split,
                                    revision=self.data_revision,
                                    from_stream=self.data_from_stream,
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

        # Creates a BBPE-based tokenizer
        if self.vocab_type == 'bbpe':
            return BBPETokenizer(tokenizer_path=self.tokenizer_path,
                                 token_config_path=self.token_config_path,
                                 min_freq=self.vocab_min_freq,
                                 vocab_size=self.vocab_size,
                                 model_max_length=self.tokenizer_max_length)

        # Creates a GPT-2-based tokenizer
        if self.vocab_type == 'gpt2':
            return GPT2Tokenizer(tokenizer_path=self.tokenizer_path,
                                 token_config_path=self.token_config_path)
        
        raise NotImplementedError(f'vocab_type: {self.vocab_type} not supported yet.')

    def _encode(self,
                tokenizer: ArchaiPreTrainedTokenizer,
                token_config: TokenConfig,
                dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]
                ) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        def _shuffle_and_select(d):
            if self.data_random_seed > 0:
                d = d.shuffle(self.data_random_seed)
            if self.data_n_samples > 0:
                d = d.select(range(self.data_n_samples))

            return d

        # Checks if dataset is a DatasetDict (usually contains multiple splits)
        if isinstance(dataset, DatasetDict):
            for split, data in dataset.items():
                dataset[split] = _shuffle_and_select(data)
        else:
            dataset = _shuffle_and_select(dataset)

        def _preprocess_dataset(x):
            return {self.data_input_column_name: token_config.pre_process_batch(x[self.data_input_column_name])}

        def _merge_dataset(x):
            return {self.data_input_column_name: [''.join(x[self.data_input_column_name])]}

        def _tokenize_dataset(x):
            return tokenizer(x[self.data_input_column_name],
                             truncation=self.data_truncate,
                             padding=self.data_padding)

        # Pre-processes the dataset
        dataset = dataset.map(lambda x: _preprocess_dataset(x), batched=True)

        # Checks if dataset should be merged into a contiguous sample
        if self.data_contiguous:
            dataset = dataset.map(lambda x: _merge_dataset(x), batched=True, batch_size=None)

        # Tokenizes the dataset and changes its format to PyTorch
        dataset = dataset.map(lambda x: _tokenize_dataset(x), batched=True, remove_columns=self.data_input_column_name)
        dataset = dataset.with_format(type='torch')

        return dataset

    def load(self) -> None:
        dataset = self._load_dataset()
        
        # Tokenizer should be re-trained it it has not been trained yet
        # or if its cache should be refreshed
        if not self.is_tokenizer_trained or self.tokenizer_refresh_cache:
            tokenizer = self._create_tokenizer()
            tokenizer.train(dataset, column_name=self.data_input_column_name)

        # Loads the tokenizer and encodes the dataset
        tokenizer = ArchaiPreTrainedTokenizer.from_file(self.tokenizer_path, self.token_config_path)
        encoded_dataset = self._encode(tokenizer, tokenizer.token_config, dataset)

        # Do not forget to attach them as attributes
        self.dataset = encoded_dataset
        self.tokenizer = tokenizer
        self.token_config = tokenizer.token_config
