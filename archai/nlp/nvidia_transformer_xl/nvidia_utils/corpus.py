"""Corpus-based utilities that encodes vocabularies.
"""

import logging
import os
from typing import Optional, Tuple, Union

import numpy as np
import torch

from archai.common import utils
from archai.nlp.nvidia_transformer_xl.nvidia_utils.lm_iterators import LMOrderedIterator, LMShuffledIterator
from archai.nlp.tokenizer_utils.bbpe_vocab import BbpeVocab
from archai.nlp.tokenizer_utils.gpt2_vocab import Gpt2Vocab
from archai.nlp.tokenizer_utils.vocab_base import VocabBase
from archai.nlp.tokenizer_utils.word_vocab import WordVocab


class Corpus:
    """Provides an entry point that receives a dataset, a type of vocabulary and
    encodes raw text into tokens.

    """

    def __init__(self,
                 datadir: str,
                 dataset: str,
                 vocab_type: str,
                 cachedir: Optional[str] = None,
                 vocab_size: Optional[int] = None,
                 refresh_cache: Optional[bool] = False) -> None:
        """Initializes common properties and the correct paths for holding cache-related files.

        Args:
            datadir: Folder where data should be stored.
            dataset: Dataset to be transformed into corpus.
            vocab_type: Type of vocabulary.
            cachedir: Folder where cache should be stored.
            vocab_size: Size of the vocabulary.
            refresh_cache: Whether cache should be refreshed or not.

        """

        self.datadir = datadir
        self.dataset = dataset
        self.vocab_type = vocab_type
        self.vocab_size = vocab_size

        # Cache folder
        # Based on dataset + vocab_type + vocab_size
        self.corpus_cache_dir = cachedir or os.path.join(datadir, 'cache')
        self.corpus_cache_dir = utils.full_path(os.path.join(self.corpus_cache_dir,
                                                             dataset,
                                                             vocab_type,
                                                             str(vocab_size)),
                                                create=True)

        # Cached numpy files
        self.train_cache_filepath = os.path.join(self.corpus_cache_dir, 'train.npy')
        self.valid_cache_filepath = os.path.join(self.corpus_cache_dir, 'valid.npy')
        self.test_cache_filepath = os.path.join(self.corpus_cache_dir, 'test.npy')

        # Cached tokenizer files
        self._vocab_cache_dir = os.path.join(self.corpus_cache_dir, 'vocab')
        self.refresh_cache = refresh_cache

        if refresh_cache:
            logging.warn('Refreshing all cache ...')

        self._clear()

    def train_and_encode(self) -> None:
        """Trains a new corpus and encodes all available sets (training, validation and/or testing).

        """

        logging.info(f'Producing corpus cache for dataset {self.dataset}, vocab_type {self.vocab_type} and vocab_size {self.vocab_size} ...')

        self.vocab = self._create_train_vocab(self.datadir,
                                              self.dataset,
                                              self.vocab_type,
                                              self._vocab_cache_dir,
                                              self.vocab_size,
                                              self.refresh_cache)

        self.train, self.valid, self.test = self._get_encoded_files(self.vocab,
                                                                    self.datadir,
                                                                    self.dataset)

    def load(self) -> None:
        """Loads cached files.

        """

        # Ensures that we are loading the tokenizer cache as well
        self.vocab = Corpus._create_vocab(self.datadir,
                                          self.dataset,
                                          self.vocab_type,
                                          self._vocab_cache_dir,
                                          vocab_size=self.vocab_size)

        # Checks whether cached files already exists
        cache_filepath = os.path.exists(self.train_cache_filepath) \
            and os.path.exists(self.valid_cache_filepath) \
            and os.path.exists(self.test_cache_filepath)

        if not self.refresh_cache and cache_filepath and self.vocab.is_trained():
            logging.info(f'Found existing cache for dataset {self.dataset}. Loading from {self.train_cache_filepath} ...')

            self.vocab.load()

            self.train = torch.from_numpy(np.load(self.train_cache_filepath))
            self.valid = torch.from_numpy(np.load(self.valid_cache_filepath))
            self.test = torch.from_numpy(np.load(self.test_cache_filepath))

            return True
        else:
            # Clears cache and allows for a new refresh
            self._clear()

            utils.delete_file(self.train_cache_filepath)
            utils.delete_file(self.valid_cache_filepath)
            utils.delete_file(self.test_cache_filepath)

            return False

    def _clear(self) -> None:
        """Clears the cache.

        """

        self.train = self.valid = self.test = self.vocab = None

    def save(self) -> None:
        """Saves to cached files.

        """

        assert self.vocab is not None and self.vocab.is_trained()

        # Ensures dataset cache is saved
        np.save(self.train_cache_filepath, self.train.numpy())
        np.save(self.valid_cache_filepath, self.valid.numpy())
        np.save(self.test_cache_filepath, self.test.numpy())

    @staticmethod
    def _create_train_vocab(datadir: str,
                            dataset: str,
                            vocab_type: str,
                            vocab_cache_dir: str,
                            vocab_size: int,
                            refresh_cache: bool) -> VocabBase:
        """Creates and trains a new vocabulary.

        Args:
            datadir: Folder where data should be stored.
            dataset: Dataset that should be used to create the vocabulary.
            vocab_type: Type of vocabulary.
            vocab_cache_dir: Folder where vocabulary cache should be stored.
            vocab_size: Size of the vocabulary.
            refresh_cache: Whether cache should be refreshed or not.

        Returns:
            (VocabBase): An instance of a pre-trained vocabulary.

        """

        vocab = Corpus._create_vocab(datadir,
                                     dataset,
                                     vocab_type,
                                     vocab_cache_dir,
                                     vocab_size=vocab_size)

        Corpus._train_vocab(vocab,
                            datadir,
                            dataset,
                            vocab_type,
                            vocab_cache_dir,
                            vocab_size,
                            refresh_cache)

        return vocab

    @staticmethod
    def _get_encoded_files(vocab: VocabBase,
                           datadir: str,
                           dataset: str) -> Tuple[torch.Tensor, ...]:
        """Gathers pre-encoded files.

        Args:
            vocab: An instance of the vocabulary.
            datadir: Folder where data was stored.
            dataset: Dataset that should be used to retrieve the files.

        Returns:
            (Tuple[torch.Tensor, ...]): Tensors holding encoded data from input files.

        """

        train_filename, test_filename, valid_filename = Corpus._dataset_filenames(dataset)

        train = vocab.encode_file(os.path.join(datadir, train_filename))
        valid = vocab.encode_file(os.path.join(datadir, valid_filename))
        test = vocab.encode_file(os.path.join(datadir, test_filename))

        return (torch.LongTensor(train), torch.LongTensor(valid), torch.LongTensor(test))

    @staticmethod
    def _create_vocab(datadir: str,
                      dataset: str,
                      vocab_type: str,
                      vocab_cache_dir: str,
                      vocab_size: Optional[int] = None) -> VocabBase:
        """Creates a new vocabulary.

        Args:
            datadir: Folder where data should be stored.
            dataset: Dataset that should be used to create the vocabulary.
            vocab_type: Type of vocabulary.
            vocab_cache_dir: Folder where vocabulary cache should be stored.
            vocab_size: Size of the vocabulary.

        Returns:
            (VocabBase): An instance of a new vocabulary.

        """

        if vocab_type == 'word':
            # Word-based file is composed of raw text of symbols, one per line
            special, lower_case, vocab_file = [], True, None

            if dataset in ['wt103', 'wt2', 'olx']:
                # TODO: we probably don't need special or could be done differently
                special, lower_case = ['<eos>'], False
            elif dataset == 'ptb':
                special, lower_case = ['<eos>'], True
            elif dataset == 'lm1b':
                special, lower_case, vocab_file = [], False, os.path.join(datadir, '1b_word_vocab.txt')
            elif dataset in ['enwik8', 'text8']:
                pass
            else:
                raise RuntimeError(f'Dataset not yet fully supported: {dataset}')

            # `<S>` is added for double end-of-sentence
            # `<unk>` is a rare token in corpus with frequency < 3
            special += ['<unk>', '<S>']
            add_eos = dataset in ['ptb', 'wt2', 'wt103', 'lm1b', 'olx']
            vocab = WordVocab(save_path=vocab_cache_dir,
                              vocab_size=vocab_size,
                              special=special,
                              lower_case=lower_case,
                              add_eos=add_eos)

        elif vocab_type == 'bbpe':
            # Default `vocab_size` for GPT-2 is 50257
            vocab = BbpeVocab(save_path=vocab_cache_dir,
                              vocab_size=vocab_size or 50257)

        elif vocab_type == 'gpt2':
            # Default `vocab_size` for GPT-2 is 50257
            vocab = Gpt2Vocab(save_path=vocab_cache_dir,
                              vocab_size=vocab_size or 50257)
        else:
            raise RuntimeError(f'Vocab type not yet fully supported: {vocab_type}')

        return vocab

    @staticmethod
    def _dataset_filenames(dataset: str) -> Tuple[str, ...]:
        """Gathers proper file names from available datasets.

        Args:
            dataset: Dataset to retrieve file names.

        Returns:
            (Tuple[str, ...]): File names for training, validation and testing sets.

        """

        train_filename, test_filename, valid_filename = 'train.txt', 'test.txt', 'valid.txt'

        if dataset in ['wt2', 'wt103', 'olx']:
            train_filename = 'wiki.train.tokens'
            test_filename = 'wiki.test.tokens'
            valid_filename = 'wiki.valid.tokens'

        return train_filename, test_filename, valid_filename

    @staticmethod
    def _train_vocab(vocab: VocabBase,
                     datadir: str,
                     dataset: str,
                     vocab_type: str,
                     vocab_cache_dir: str,
                     vocab_size: int,
                     refresh_cache: bool) -> None:
        """Trains a new vocabulary.

        Args:
            vocab: Instance of a created vocabulary.
            datadir: Folder where data should be stored.
            dataset: Dataset that should be used to train the vocabulary.
            vocab_type: Type of vocabulary.
            vocab_cache_dir: Folder where vocabulary cache is stored.
            vocab_size: Size of the vocabulary.
            refresh_cache: Whether cache should be refreshed or not.

        """

        # Checks whether cached vocabulary exists or not
        if refresh_cache or not vocab.is_trained():
            train_filename, _, _ = Corpus._dataset_filenames(dataset)

            logging.info('Training vocab ...')
            vocab.train([os.path.join(datadir, train_filename)])
            logging.info('Training finished.')
        else:
            vocab.load()
            logging.info(f'Vocab cache found and loaded for type {vocab_type} and size {vocab_size} from {vocab_cache_dir}.')

    def get_iterator(self,
                     split: str,
                     *args,
                     **kwargs) -> Union[LMOrderedIterator, LMShuffledIterator]:
        """Gets a language modeling iterator.

        Args:
            split: Name of split to retrieve the iterator from.

        Returns:
            (Union[LMOrderedIterator, LMShuffledIterator]): Language modeling iterator.

        """

        if split == 'train':
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
                data_iter = LMOrderedIterator(self.train, *args, **kwargs)

                # TODO: add support for iterator over `lm1b` training set
                # elif self.dataset == 'lm1b':
                #     kwargs['shuffle'] = True
                #     data_iter = LMMultiFileIterator(self.train, self.vocab, *args, **kwargs)

            else:
                raise RuntimeError(f'Dataset not yet fully supported: {self.dataset}')

        elif split in ['valid', 'test']:
            data = self.valid if split == 'valid' else self.test

            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
                data_iter = LMOrderedIterator(data, *args, **kwargs)

            # TODO: add support for iterator over `lm1b` valid and testing set
            # elif self.dataset == 'lm1b':
            #     data_iter = LMShuffledIterator(data, *args, **kwargs)

            else:
                raise RuntimeError(f'Dataset not yet fully supported: {self.dataset}')

        else:
            raise RuntimeError(f'Split not yet fully supported: {split}')

        return data_iter
