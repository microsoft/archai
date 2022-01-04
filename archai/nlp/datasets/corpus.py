# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Corpus-related utilities that defines input data.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from archai.common import utils
from archai.nlp.datasets.lm_iterators import (LMOrderedIterator,
                                              LMShuffledIterator)
from archai.nlp.datasets.tokenizer_utils.bbpe_vocab import BbpeVocab
from archai.nlp.datasets.tokenizer_utils.gpt2_vocab import Gpt2Vocab
from archai.nlp.datasets.tokenizer_utils.vocab_base import VocabBase
from archai.nlp.datasets.tokenizer_utils.word_vocab import WordVocab


@dataclass
class DataFileStats:
    """Provides file statistics for datasets.

    """

    filepath: str
    line_count: int = 0
    word_count: int = 0
    char_count: int = 0

class Corpus:
    """Implements the Corpus.

    """

    def __init__(self,
                 datadir: str,
                 dataset: str,
                 vocab_type: str,
                 cachedir: str,
                 vocab_size: Optional[int] = None,
                 refresh_cache: Optional[bool] = False) -> None:
        """Overrides initialization method.

        Args:
            datadir: Directory of data.
            dataset: Dataset identifier.
            vocab_type: Type of vocabulary.
            cachedir: Directory of cached data.
            vocab_size: Size of vocabulary.
            refresh_cache: Whether to refresh the cache or not.

        """

        self.datadir = datadir
        self.dataset = dataset
        self.vocab_type = vocab_type
        self.vocab_size =  vocab_size

        # where we maintain the cache for the corpus based on dataset+vocab_type+vocab_size
        self.corpus_cache_dir = cachedir
        self.corpus_cache_dir = utils.full_path(os.path.join(self.corpus_cache_dir, f'{dataset}',f'{vocab_type}',f'{vocab_size}'), create=True)

        # where dataset npy files are cached
        self.train_cache_filepath = os.path.join(self.corpus_cache_dir, 'train.npy')
        self.valid_cache_filepath = os.path.join(self.corpus_cache_dir, 'valid.npy')
        self.test_cache_filepath = os.path.join(self.corpus_cache_dir, 'test.npy')

        # where tokenizer files will be cached
        self._vocab_cache_dir = os.path.join(self.corpus_cache_dir, 'vocab')
        self.refresh_cache = refresh_cache

        if refresh_cache:
            logging.warn('refresh_cache=True, all cache will be refreshed')

        self._clear()

    def train_and_encode(self) -> None:
        """Trains a vocabulary and encodes supplied data.

        """

        logging.info(f'Producing corpus cache for dataset {self.dataset}, vocab_type{self.vocab_type}, vocab_size {self.vocab_size}...')

        self.vocab = self._create_train_vocab()

        self.train, self.valid, self.test = self._get_encoded_files()

        logging.info(f'Sizes for train: {self.train.size(0)}, valid: {self.valid.size(0)}, test: {self.test.size(0)}')


    def load(self) -> None:
        """Loads a pre-trained vocabulary and cached data, if available.

        """

        # ensure that we have tokenizer cache as well
        self.vocab = Corpus._create_vocab(self.datadir, self.dataset,
            self.vocab_type, self._vocab_cache_dir, vocab_size=self.vocab_size)

        cache_exists = os.path.exists(self.train_cache_filepath) and os.path.exists(self.valid_cache_filepath) and os.path.exists(self.test_cache_filepath)

        # if files for all dataset npy exist then we have corpus cache
        if not self.refresh_cache and cache_exists and self.vocab is not None and self.vocab.is_trained():
            logging.info(f'Found existing cache for for dataset {self.dataset}. Loading from {self.train_cache_filepath}.')

            self.vocab.load()

            self.train = torch.from_numpy(np.load(self.train_cache_filepath))
            self.valid = torch.from_numpy(np.load(self.valid_cache_filepath))
            self.test = torch.from_numpy(np.load(self.test_cache_filepath))

            logging.info(f'Sizes for train: {self.train.size(0)}, valid: {self.valid.size(0)}, test: {self.test.size(0)}')

            return True

        else:
            logging.info(f'Clearing all cache and rebuidling it')

            self._clear()

            utils.delete_file(self.train_cache_filepath)
            utils.delete_file(self.valid_cache_filepath)
            utils.delete_file(self.test_cache_filepath)

            return False # no cache exists or refresh is needed

    def _clear(self) -> None:
        """Clears the cached corpus.

        """

        self.train = self.valid  = self.test = self.vocab = None

    def save(self) -> None:
        """Saves the corpus to cache files.

        """

        assert self.vocab is not None and self.vocab.is_trained()

        # save dataset cache
        np.save(self.train_cache_filepath, self.train.numpy())
        np.save(self.valid_cache_filepath, self.valid.numpy())
        np.save(self.test_cache_filepath, self.test.numpy())

    def _create_train_vocab(self) -> VocabBase:
        """Creates and trains a new vocabulary.

        Returns:
            (VocabBase): Trained vocabulary.

        """

        self.vocab = Corpus._create_vocab(self.datadir, self.dataset, self.vocab_type,
                                          self._vocab_cache_dir, vocab_size=self.vocab_size)
        self._train_vocab()

        return self.vocab

    @staticmethod
    def _get_file_stats(filepath: str) -> DataFileStats:
        """Gets file statistics.

        Args:
            filepath: File to be analyzed.

        Returns:
            (DataFileStats): File statistics.

        """

        stats = DataFileStats(filepath)

        with open(filepath, 'r', encoding="utf-8") as f:
            for line in f:
                stats.line_count += 1
                stats.char_count += len(line)
                stats.word_count += len(line.split())

        return stats

    def file_stats(self) -> Tuple[DataFileStats, DataFileStats, DataFileStats]:
        """Gets file statistics for training, validation and testing sets.

        Returns:
            (Tuple[DataFileStats, DataFileStats, DataFileStats]): Training, validation and
                testing set file statistics.

        """

        train_filepath, valid_filepath, test_filepath = self._dataset_filepaths()
        
        return (Corpus._get_file_stats(train_filepath), \
                Corpus._get_file_stats(valid_filepath), \
                Corpus._get_file_stats(test_filepath))

    def _get_encoded_files(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gets encoded files for training, validation and testing sets.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Training, validation and
                testing set encoded files.

        """

        train_filepath, valid_filepath, test_filepath = self._dataset_filepaths()

        train = self.vocab.encode_file(train_filepath)
        valid = self.vocab.encode_file(valid_filepath)
        test = self.vocab.encode_file(test_filepath)

        return (train, valid, test)

    @staticmethod
    def _create_vocab(datadir: str,
                      dataset: str,
                      vocab_type: str,
                      vocab_cache_dir: str,
                      vocab_size: Optional[int] = None) -> VocabBase:
        """Creates a new vocabulary.

        Args:
            datadir: Directory of data.
            dataset: Dataset identifier.
            vocab_type: Type of vocabulary.
            vocab_cache_dir: Directory of cached vocabulary.
            vocab_size: Size of vocabulary.

        Returns:
            (VocabBase): Vocabulary.

        """

        if vocab_type == 'word':
            # '<S>' is added for double eos and <unk> is rare token in corpus with freq < 3
            bos_token, eos_token, lower_case, vocab_file = None, '<eos>', False, None # vocab file is text file of symbols, one per line

            if dataset in ['wt103', 'wt2'] or dataset.startswith('olx_'):
                pass

            elif dataset == 'ptb':
                lower_case = True

            elif dataset == 'lm1b':
                bos_token, eos_token, vocab_file = '<S>', '<S>', os.path.join(datadir, '1b_word_vocab.txt')

            elif dataset in ['enwik8', 'text8']:
                eos_token, lower_case = None, True

            else:
                raise RuntimeError(f'dataset {dataset} is not recognized to produce vocab')

            vocab = WordVocab(save_path=vocab_cache_dir, vocab_size=vocab_size,
                              bos_token=bos_token, eos_token=eos_token,
                              lower_case=lower_case)

        elif vocab_type == 'bbpe':
            vocab = BbpeVocab(save_path=vocab_cache_dir, vocab_size=vocab_size or 50257) # default vocab size for GPT-2 is 50257

        elif vocab_type == 'gpt2':
            vocab = Gpt2Vocab(save_path=vocab_cache_dir, vocab_size=vocab_size or 50257) # default vocab size for GPT-2 is 50257

        else:
            raise RuntimeError(f'Unsupported vocab type: {vocab_type}')

        return vocab

    def _dataset_filepaths(self) -> Tuple[str, str, str]:
        """Gets file paths for training, validation and testing sets.

        Returns:
            (Tuple[str, str, str]): Training, validation and
                testing set file paths.

        """

        train_filename, valid_filename, test_filename = 'train.txt', 'valid.txt', 'test.txt'

        if self.dataset in ['wt2', 'wt103']:
            train_filename, valid_filename, test_filename = 'wiki.train.tokens', 'wiki.valid.tokens', 'wiki.test.tokens'

        return (os.path.join(self.datadir, train_filename),
                os.path.join(self.datadir, valid_filename),
                os.path.join(self.datadir, test_filename))

    def _train_vocab(self) -> None:
        """Trains vocabulary if it has not been trained already.

        """

        if self.refresh_cache or not self.vocab.is_trained(): # if vocab cache does not exist
            train_filepath, valid_filepath, test_filepath = \
                self._dataset_filepaths()

            logging.info('Training vocab...')

            self.vocab.train([train_filepath])

            logging.info('Finished training vocab.')

        else:
            self.vocab.load()

            logging.info(f'Vocab cache found and loaded for type {self.vocab_type} and size {self.vocab_size} from {self._vocab_cache_dir}.')

    def get_iterator(self,
                     split: str,
                     batch_size: int,
                     tgt_len: int,
                     device: str,
                     ext_len: int,
                     mem_len: Optional[int] = None) -> Union[LMOrderedIterator, LMShuffledIterator]:
        """Gets an iterator.

        Args:
            split: Type of split (train, val or test).
            batch_size: Size of batch.
            tgt_len: Length of target sequences.
            device: Device to be used.
            ext_len: Length of the extended context.
            mem_len: Length of the memory.

        Returns:
            (Union[LMOrderedIterator, LMShuffledIterator]): Language Modeling iterator.

        """

        if split == 'train':
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8'] or self.dataset.startswith('olx_'):
                data_iter = LMOrderedIterator(self.train, batch_size, tgt_len,
                                              device=device, ext_len=ext_len, mem_len=mem_len)
            # elif self.dataset == 'lm1b':
            #     kwargs['shuffle'] = True
            #     data_iter = LMMultiFileIterator(self.train, self.vocab, *args, **kwargs)
            else:
                raise RuntimeError(f'Dataset not yet fully supported: {self.dataset}')

        elif split in ['valid', 'test']:
            data = self.valid if split == 'valid' else self.test
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8'] or self.dataset.startswith('olx_'):
                data_iter = LMOrderedIterator(data, batch_size, tgt_len,
                                              device=device, ext_len=ext_len, mem_len=mem_len)
            elif self.dataset == 'lm1b':
                data_iter = LMShuffledIterator(data, batch_size, tgt_len,
                                              device=device, ext_len=ext_len)
            else:
                raise RuntimeError(f'Dataset not yet fully supported: {self.dataset}')

        else:
            raise RuntimeError(f'split not supported: {split}')

        return data_iter
