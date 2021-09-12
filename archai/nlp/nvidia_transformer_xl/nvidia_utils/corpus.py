import os
import glob
from typing import Optional, Tuple
import logging

import numpy as np
import torch

from archai.common import utils
from archai.nlp.tokenizer_utils.vocab_base import VocabBase
from archai.nlp.tokenizer_utils.word_vocab import WordVocab
from archai.nlp.tokenizer_utils.gpt2_vocab import Gpt2Vocab
from archai.nlp.nvidia_transformer_xl.nvidia_utils.lm_iterators import LMMultiFileIterator, LMOrderedIterator, LMShuffledIterator

class Corpus:
    def __init__(self, datadir:str, dataset:str, vocab_type:str, cachedir:str,
                 vocab_size:Optional[int]=None):
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

        self._clear()

    def train_and_encode(self):
        logging.info(f'Producing corpus cache for dataset {self.dataset}, vocab_type{self.vocab_type}, vocab_size {self.vocab_size}...')

        self.vocab = self._create_train_vocab(self.datadir, self.dataset, self.vocab_type,
                                            self._vocab_cache_dir, vocab_size=self.vocab_size)

        self.train, self.valid, self.test = self._get_encoded_files(
            self.vocab, self.datadir, self.dataset)

    def load(self):
        # if files for all dataset npy exist then we have corpus cache
        if os.path.exists(self.train_cache_filepath) and os.path.exists(self.valid_cache_filepath) and os.path.exists(self.test_cache_filepath):
            logging.info(f'Found existing cache for for dataset {self.dataset}. Loading from {self.train_cache_filepath}.')

            # ensure that we have tokenizer cache as well
            self.vocab = Corpus._create_vocab(self.datadir, self.dataset,
                self.vocab_type, self._vocab_cache_dir, vocab_size=self.vocab_size)
            vocab_exist = self.vocab.load()
            assert vocab_exist

            self.train = torch.from_numpy(np.load(self.train_cache_filepath))
            self.valid = torch.from_numpy(np.load(self.valid_cache_filepath))
            self.test = torch.from_numpy(np.load(self.test_cache_filepath))

            return True
        else:
            self._clear()
            return False # no cache exists

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
                      vocab_size:Optional[int]=None)->VocabBase:
        vocab = Corpus._create_vocab(datadir, dataset, vocab_type,
                                     vocab_cache_dir, vocab_size=vocab_size)
        Corpus._train_vocab(vocab, datadir, dataset, vocab_type,
                                     vocab_cache_dir, vocab_size=vocab_size)

        return vocab

    @staticmethod
    def _get_encoded_files(vocab:VocabBase, datadir:str, dataset:str)->\
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        train_filename, test_filename, valid_filename = \
                    Corpus._dataset_filenames(dataset)

        train = vocab.encode_file(os.path.join(datadir, train_filename))
        valid = vocab.encode_file(os.path.join(datadir, valid_filename))
        test = vocab.encode_file(os.path.join(datadir, test_filename))

        return (torch.LongTensor(train), torch.LongTensor(valid), torch.LongTensor(test))

    @staticmethod
    def _create_vocab(datadir:str, dataset:str, vocab_type:str, vocab_cache_dir:str,
                      vocab_size:Optional[int]=None)->VocabBase:
        if vocab_type == 'word':
            special, lower_case, vocab_file = [], True, None # vocab file is text file of symbols, one per line
            if dataset in ['wt103', 'wt2']:
                special, lower_case = ['<eos>'], False
            elif dataset == 'ptb':
                special, lower_case = ['<eos>'], True
            elif dataset == 'lm1b':
                special, lower_case, vocab_file = [], False, os.path.join(datadir, '1b_word_vocab.txt')
            elif dataset in ['enwik8', 'text8']:
                pass
            else:
                raise RuntimeError(f'dataset {dataset} is not recognized to produce vocab')

            special += ['<unk>', '<S>'] # '<S>' is added for double eos and <unk> is rare token in corpus with freq < 3
            add_eos = dataset in ['ptb', 'wt2', 'wt103', 'lm1b']
            vocab = WordVocab(save_path=vocab_cache_dir, vocab_size=vocab_size, special=special, lower_case=lower_case, add_eos=add_eos)
        elif vocab_type == 'bpe':
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
                    vocab_cache_dir:str, vocab_size:Optional[int]=None)->None:

        if not vocab.is_trained(): # if vocab cache does not exist
            train_filename, test_filename, valid_filename = \
                Corpus._dataset_filenames(dataset)

            vocab.train([os.path.join(datadir, train_filename)])
        else:
            vocab.load()
            logging(f'Vocab cache found and loaded for type {vocab_type} and size {vocab_size} from {vocab_cache_dir}.')

    def get_iterator(self, split, *args, **kwargs):
        if split == 'train':
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
                data_iter = LMOrderedIterator(self.train, *args, **kwargs)
            # elif self.dataset == 'lm1b':
            #     kwargs['shuffle'] = True
            #     data_iter = LMMultiFileIterator(self.train, self.vocab, *args, **kwargs)
            else:
                raise RuntimeError(f'Dataset not yet fully supported: {self.dataset}')

        elif split in ['valid', 'test']:
            data = self.valid if split == 'valid' else self.test
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
                data_iter = LMOrderedIterator(data, *args, **kwargs)
            elif self.dataset == 'lm1b':
                data_iter = LMShuffledIterator(data, *args, **kwargs)
            else:
                raise RuntimeError(f'Dataset not yet fully supported: {self.dataset}')

        else:
            raise RuntimeError(f'split not supported: {split}')

        return data_iter