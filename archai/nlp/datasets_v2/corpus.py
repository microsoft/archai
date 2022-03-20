# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

import os
import json

from typing import Any, Tuple, Union, Dict, List, Optional
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

from transformers import PreTrainedTokenizerFast
from archai.nlp.datasets_v2.dataset_loader import load_file_dataset, load_hub_dataset
from archai.nlp.datasets_v2.tokenizer_utils.token_config import TokenConfig
from archai.nlp.datasets_v2.tokenizer_utils.vocab_base import Vocab
from archai.nlp.datasets_v2.tokenizer_utils.word_vocab import WordVocab


class Corpus:
    def __init__(self,
                 data_dir: str,
                 cache_dir: str,
                 data_config_name: Optional[str] = None,
                 data_type: Optional[str] = 'file',
                 data_files: Optional[Union[Dict[str, Any], List[str]]] = None,
                 data_split: Optional[List[str]] = None,
                 data_revision: Optional[str] = None,
                 data_features: Optional[List[str]] = None,
                 data_column_name: Optional[str] = None,
                 from_stream: Optional[bool] = False,
                 refresh_cache: Optional[bool] = False,
                 vocab_type: Optional[str] = 'word',
                 vocab_size: Optional[int] = 10000) -> None:
        """
        """

        # Dataset-related attributes
        self.data_dir = data_dir
        self.data_config_name = data_config_name
        self.data_type = data_type
        self.data_files = data_files
        self.data_split = data_split
        self.data_revision = data_revision
        self.data_features = data_features
        self.data_column_name = data_column_name

        # Cache/streaming-related attributes
        self.cache_dir = os.path.join(cache_dir, vocab_type, str(vocab_size))
        self.from_stream = from_stream
        self.refresh_cache = refresh_cache

        # Vocabulary-related attributes
        self.vocab_type = vocab_type
        self.vocab_size = vocab_size
        self.vocab_path = os.path.join(self.cache_dir, 'tokenizer.json')
        self.vocab_config_path = os.path.join(self.cache_dir, 'token_config.json')

    @property
    def is_vocab_trained(self) -> bool:
        return os.path.exists(self.vocab_path) and os.path.exists(self.vocab_config_path)

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

        # Loads dataset from external Huggingface's datasets hub
        elif self.data_type == 'hub':
            return load_hub_dataset(self.data_dir,
                                    self.data_config_name,
                                    self.cache_dir,
                                    split=self.data_split,
                                    revision=self.data_revision,
                                    from_stream=self.from_stream,
                                    refresh_cache=self.refresh_cache)
        else:
            raise NotImplementedError()

    def _create_vocab(self) -> Vocab:
        # Creates a word-based vocabulary (tokenizer)
        if self.vocab_type == 'word':
            return WordVocab()
        else:
            raise NotImplementedError()

    def _load_vocab(self) -> Tuple[TokenConfig, PreTrainedTokenizerFast]:
        # Attempts to load the token's configuration because it will be missed
        # when creating the PreTrainedTokenizerFast from file
        try:
            with open(self.vocab_config_path, 'r') as f:
                config = json.load(f)
        except:
            raise FileNotFoundError(f'{self.vocab_config_path} could not be found.')
        
        # Attempts to load a pre-trained vocabulary (compatible with `transformers`)
        # from its pre-trained file
        try:
            return TokenConfig(**config), PreTrainedTokenizerFast(model_max_length=config['model_max_length'],
                                                                  bos_token=config['bos_token'],
                                                                  eos_token=config['eos_token'],
                                                                  unk_token=config['unk_token'],
                                                                  sep_token=config['sep_token'],
                                                                  pad_token=config['pad_token'],
                                                                  cls_token=config['cls_token'],
                                                                  mask_token=config['mask_token'],
                                                                  tokenizer_file=self.vocab_path)
        except:
            raise FileNotFoundError(f'{self.vocab_path} could not be found.')

    def encode(self,
               config: TokenConfig,
               vocab: PreTrainedTokenizerFast,
               dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]
               ) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        def _apply_tokenization(x):
            return vocab(config.pre_process(x[self.data_column_name]))

        # Encodes the dataset by applying the pre-processing and tokenization
        encoded_dataset = dataset.map(lambda x: _apply_tokenization(x), batched=True)

        return encoded_dataset

    def load(self) -> None:
        # Loads the dataset
        dataset = self._load_dataset()

        # If vocabulary has not been trained or
        # if cache should be refreshed
        if not self.is_vocab_trained or self.refresh_cache:
            vocab = self._create_vocab()
            vocab.train(self.vocab_path,
                        dataset,
                        column_name=self.data_column_name)

        # Loads pre-trained vocab and encodes the dataset
        config, vocab = self._load_vocab()
        encoded_dataset = self.encode(config, vocab, dataset)

        # Attaches them as attributes
        self.dataset = encoded_dataset
        self.config = config
        self.vocab = vocab
            
        
def get_corpus(data_dir: str,
               cache_dir: str,
               data_config_name: Optional[str] = None,
               data_type: Optional[str] = 'file',
               data_files: Optional[Union[Dict[str, Any], List[str]]] = None,
               data_split: Optional[List[str]] = None,
               data_revision: Optional[str] = None,
               data_features: Optional[List[str]] = None,
               data_column_name: Optional[str] = 'text',
               from_stream: Optional[bool] = False,
               refresh_cache: Optional[bool] = False,
               vocab_type: Optional[str] = 'word',
               vocab_size: Optional[int] = 10000) -> Corpus:
    corpus = Corpus(data_dir,
                    cache_dir,
                    data_config_name=data_config_name,
                    data_type=data_type,
                    data_files=data_files,
                    data_split=data_split,
                    data_revision=data_revision,
                    data_features=data_features,
                    data_column_name=data_column_name,
                    from_stream=from_stream,
                    refresh_cache=refresh_cache,
                    vocab_type=vocab_type,
                    vocab_size=vocab_size)
    corpus.load()

    return corpus
