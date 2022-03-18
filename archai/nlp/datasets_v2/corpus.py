# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

import os
from typing import Any, Union, Dict, List, Optional
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

from transformers import PreTrainedTokenizerFast
from archai.nlp.datasets_v2.dataset_loader import load_file_dataset, load_hub_dataset
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

        # Cache/streaming-related attributes
        self.cache_dir = os.path.join(cache_dir, vocab_type, str(vocab_size))
        self.from_stream = from_stream
        self.refresh_cache = refresh_cache

        # Vocabulary-related attributes
        self.vocab_type = vocab_type
        self.vocab_size = vocab_size
        self.vocab_path = os.path.join(self.cache_dir, 'tokenizer.json')
    

    def _load_dataset(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        """
        """

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

    @property
    def is_vocab_trained(self) -> bool:
        """
        """

        return os.path.exists(self.vocab_path)

    def _create_vocab(self) -> Vocab:
        """
        """
        
        # Creates a word-based vocabulary (tokenizer)
        if self.vocab_type == 'word':
            return WordVocab()
        else:
            raise NotImplementedError()

    def _load_vocab(self) -> PreTrainedTokenizerFast:
        """
        """

        # Attempts to load a pre-trained vocabulary (compatible with `transformers`)
        # from its pre-trained file
        try:
            
            return PreTrainedTokenizerFast(tokenizer_file=self.vocab_path)
        except:
            raise FileNotFoundError(f'{self.vocab_path} could not be found.')

    def encode(self,
               vocab: PreTrainedTokenizerFast,
               dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]
               ) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        """
        """

        def _apply_tokenization(x: str) -> str:
            return vocab(x['text'])

        # Encodes the dataset by applying the tokenizer's tokenization
        encoded_dataset = dataset.map(lambda x: _apply_tokenization(x), batched=True)

        return encoded_dataset

    def load(self) -> None:
        """
        """

        # Loads the dataset
        dataset = self._load_dataset()

        if not self.is_vocab_trained or self.refresh_cache:
            # Creates and trains a new vocab
            vocab = self._create_vocab()
            vocab.train(dataset)

        # Loads pre-trained vocab and encodes the dataset
        vocab = self._load_vocab()
        encoded_dataset = self.encode(vocab, dataset)

        # Attaches them as attributes
        self.dataset = encoded_dataset
        self.vocab = vocab
            
        
def get_corpus(data_dir: str,
               cache_dir: str,
               data_config_name: Optional[str] = None,
               data_type: Optional[str] = 'file',
               data_files: Optional[Union[Dict[str, Any], List[str]]] = None,
               data_split: Optional[List[str]] = None,
               data_revision: Optional[str] = None,
               data_features: Optional[List[str]] = None,
               from_stream: Optional[bool] = False,
               refresh_cache: Optional[bool] = False,
               vocab_type: Optional[str] = 'word',
               vocab_size: Optional[int] = 10000) -> Corpus:
    """
    """

    corpus = Corpus(data_dir,
                    cache_dir,
                    data_config_name=data_config_name,
                    data_type=data_type,
                    data_files=data_files,
                    data_split=data_split,
                    data_revision=data_revision,
                    data_features=data_features,
                    from_stream=from_stream,
                    refresh_cache=refresh_cache,
                    vocab_type=vocab_type,
                    vocab_size=vocab_size)
    corpus.load()

    return corpus
