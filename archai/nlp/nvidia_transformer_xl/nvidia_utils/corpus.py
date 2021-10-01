import os
import glob
from typing import Optional, Tuple
import logging

import numpy as np
import torch

from archai.common import utils
from archai.nlp.tokenizer_utils.vocab_base import VocabBase
from archai.nlp.tokenizer_utils.word_vocab import WordVocab
from archai.nlp.tokenizer_utils.bbpe_vocab import BbpeVocab
from archai.nlp.tokenizer_utils.gpt2_vocab import Gpt2Vocab

from archai.nlp.nvidia_transformer_xl.nvidia_utils.lm_iterators import LMMultiFileIterator, LMOrderedIterator, LMShuffledIterator

class Corpus:
    def __init__(self, datadir:str, dataset:str, vocab_type:str, cachedir:str,
                 vocab_size:Optional[int]=None, refresh_cache=False):
        self.datadir = datadir
        self.dataset = dataset
        self.vocab_type = vocab_type
        self.vocab_size =  vocab_size

        # where we maintain the cache for the corpus based on dataset+vocab_type+vocab_size
        self.corpus_cache_dir = cachedir or os.path.join(datadir, 'cache')
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

    def train_and_encode(self):
        logging.info(f'Producing corpus cache for dataset {self.dataset}, vocab_type{self.vocab_type}, vocab_size {self.vocab_size}...')

        self.vocab = self._create_train_vocab(self.datadir, self.dataset, self.vocab_type,
                                            self._vocab_cache_dir, self.vocab_size, self.refresh_cache)

        self.train, self.valid, self.test = self._get_encoded_files(
            self.vocab, self.datadir, self.dataset)

    def load(self):
        # if files for all dataset npy exist then we have corpus cache
        if not self.refresh_cache and os.path.exists(self.train_cache_filepath) and os.path.exists(self.valid_cache_filepath) and os.path.exists(self.test_cache_filepath) and self.vocab is not None and self.vocab.is_trained():
            logging.info(f'Found existing cache for for dataset {self.dataset}. Loading from {self.train_cache_filepath}.')

            # ensure that we have tokenizer cache as well
            self.vocab = Corpus._create_vocab(self.datadir, self.dataset,
                self.vocab_type, self._vocab_cache_dir, vocab_size=self.vocab_size)
            self.vocab.load()

            self.train = torch.from_numpy(np.load(self.train_cache_filepath))
            self.valid = torch.from_numpy(np.load(self.valid_cache_filepath))
            self.test = torch.from_numpy(np.load(self.test_cache_filepath))

            return True
        else:
            self._clear()
            utils.delete_file(self.train_cache_filepath)
            utils.delete_file(self.valid_cache_filepath)
            utils.delete_file(self.test_cache_filepath)
            return False # no cache exists or refresh is needed

    def _clear(self)->None:
        self.train = self.valid  = self.test = self.vocab = None

    def save(self):
        assert self.vocab is not None and self.vocab.is_trained()

        # save dataset cache
        np.save(self.train_cache_filepath, self.train.numpy())
        np.save(self.valid_cache_filepath, self.valid.numpy())
        np.save(self.test_cache_filepath, self.test.numpy())

    @staticmethod
    def _create_train_vocab(datadir:str, dataset:str, vocab_type:str, vocab_cache_dir:str,
                      vocab_size:Optional[int], refresh_cache:bool)->VocabBase:
        vocab = Corpus._create_vocab(datadir, dataset, vocab_type,
                                     vocab_cache_dir, vocab_size=vocab_size)
        Corpus._train_vocab(vocab, datadir, dataset, vocab_type,
                                     vocab_cache_dir, vocab_size, refresh_cache)

        return vocab

    @staticmethod
    def _get_encoded_files(vocab:VocabBase, datadir:str, dataset:str)->\
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        train_filename, test_filename, valid_filename = \
                    Corpus._dataset_filenames(dataset)

        train = vocab.encode_file(os.path.join(datadir, train_filename))
        valid = vocab.encode_file(os.path.join(datadir, valid_filename))
        test = vocab.encode_file(os.path.join(datadir, test_filename))

        return ((train, valid, test))

    @staticmethod
    def _create_vocab(datadir:str, dataset:str, vocab_type:str, vocab_cache_dir:str,
                      vocab_size:Optional[int]=None)->VocabBase:
        if vocab_type == 'word':
            # '<S>' is added for double eos and <unk> is rare token in corpus with freq < 3
            unk_token, bos_token, eos_token, lower_case, vocab_file = '<unk>', None, '<eos>', False, None # vocab file is text file of symbols, one per line
            if dataset in ['wt103', 'wt2', 'olx']:
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
                              bos_token=bos_token, eos_token=eos_token, unk_token=unk_token,
                              lower_case=lower_case)
        elif vocab_type == 'bbpe':
            vocab = BbpeVocab(save_path=vocab_cache_dir, vocab_size=vocab_size or 50257) # default vocab size for GPT-2 is 50257
        elif vocab_type == 'gpt2':
            vocab = Gpt2Vocab(save_path=vocab_cache_dir, vocab_size=vocab_size or 50257) # default vocab size for GPT-2 is 50257
        else:
            raise RuntimeError(f'Unsupported vocab type: {vocab_type}')

        return vocab

    @staticmethod
    def _dataset_filenames(dataset:str)->Tuple[str,str,str]:
        train_filename, test_filename, valid_filename = 'train.txt', 'test.txt', 'valid.txt'
        if dataset in ['wt2', 'wt103']:
            train_filename, test_filename, valid_filename = 'wiki.train.tokens', 'wiki.test.tokens', 'wiki.valid.tokens'

        return train_filename, test_filename, valid_filename

    @staticmethod
    def _train_vocab(vocab:VocabBase, datadir:str, dataset:str, vocab_type:str,
                    vocab_cache_dir:str, vocab_size:Optional[int],
                    refresh_cache:bool)->None:

        if refresh_cache or not vocab.is_trained(): # if vocab cache does not exist
            train_filename, test_filename, valid_filename = \
                Corpus._dataset_filenames(dataset)

            logging.info('Training vocab...')
            vocab.train([os.path.join(datadir, train_filename)])
            logging.info('Finished training vocab.')
        else:
            vocab.load()
            logging.info(f'Vocab cache found and loaded for type {vocab_type} and size {vocab_size} from {vocab_cache_dir}.')

    def get_iterator(self, split, *args, **kwargs):
        if split == 'train':
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8', 'olx']:
                data_iter = LMOrderedIterator(self.train, *args, **kwargs)
            # elif self.dataset == 'lm1b':
            #     kwargs['shuffle'] = True
            #     data_iter = LMMultiFileIterator(self.train, self.vocab, *args, **kwargs)
            else:
                raise RuntimeError(f'Dataset not yet fully supported: {self.dataset}')

        elif split in ['valid', 'test']:
            data = self.valid if split == 'valid' else self.test
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8', 'olx']:
                data_iter = LMOrderedIterator(data, *args, **kwargs)
            elif self.dataset == 'lm1b':
                data_iter = LMShuffledIterator(data, *args, **kwargs)
            else:
                raise RuntimeError(f'Dataset not yet fully supported: {self.dataset}')

        else:
            raise RuntimeError(f'split not supported: {split}')

        return data_iter