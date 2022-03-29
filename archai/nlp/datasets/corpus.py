import os
import glob
from typing import Optional, Tuple, Union
import logging
from dataclasses import dataclass

import numpy as np
import torch

from archai.common import utils
from archai.nlp.datasets.tokenizer_utils.vocab_base import VocabBase
from archai.nlp.datasets.tokenizer_utils.word_vocab import WordVocab
from archai.nlp.datasets.tokenizer_utils.bbpe_vocab import BbpeVocab
from archai.nlp.datasets.tokenizer_utils.gpt2_vocab import Gpt2Vocab

from archai.nlp.datasets.lm_iterators import LMMultiFileIterator, LMOrderedIterator, LMShuffledIterator


@dataclass
class DataFileStats:
    filepath:str
    line_count:int=0
    word_count:int=0
    char_count:int=0

class Corpus:
    def __init__(self, datadir:str, dataset:str, vocab_type:str, cachedir:str,
                 vocab_size:Optional[int]=None, refresh_cache=False):
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

    def train_and_encode(self):
        logging.info(f'Producing corpus cache for dataset {self.dataset}, vocab_type{self.vocab_type}, vocab_size {self.vocab_size}...')

        self.vocab = self._create_train_vocab()

        self.train, self.valid, self.test = self._get_encoded_files()
        train_size = f'{len(self.train)} files' if isinstance(self.train, list) else self.train.size(0)

        logging.info(f'Sizes for train: {train_size}, valid: {self.valid.size(0)}, test: {self.test.size(0)}')


    def load(self):
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

    def _clear(self)->None:
        self.train = self.valid  = self.test = self.vocab = None

    def save(self):
        assert self.vocab is not None and self.vocab.is_trained()

        # save dataset cache
        np.save(self.train_cache_filepath, self.train.numpy())
        np.save(self.valid_cache_filepath, self.valid.numpy())
        np.save(self.test_cache_filepath, self.test.numpy())

    def _create_train_vocab(self)->VocabBase:
        self.vocab = Corpus._create_vocab(self.datadir, self.dataset, self.vocab_type,
                                     self._vocab_cache_dir, vocab_size=self.vocab_size)
        self._train_vocab()

        return self.vocab

    @staticmethod
    def _get_file_stats(filepath:Union[str,list])->DataFileStats:
        if not isinstance(filepath, list):
            filepath = [filepath]
        
        stats = DataFileStats(filepath)
        
        for f_path in filepath:
            with open(f_path, 'r', encoding="utf-8") as f:
                for line in f:
                    stats.line_count += 1
                    stats.char_count += len(line)
                    stats.word_count += len(line.split())
        return stats

    def file_stats(self)->Tuple[DataFileStats, DataFileStats, DataFileStats]:
        train_filepath, valid_filepath, test_filepath = self._dataset_filepaths()
        return (Corpus._get_file_stats(train_filepath), \
                Corpus._get_file_stats(valid_filepath), \
                Corpus._get_file_stats(test_filepath))

    def _get_encoded_files(self)->\
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        train_filepath, valid_filepath, test_filepath = self._dataset_filepaths()

        if self.dataset == 'lm1b':
            train = train_filepath
        else:
            train = self.vocab.encode_file(train_filepath)

        valid = self.vocab.encode_file(valid_filepath)
        test = self.vocab.encode_file(test_filepath)

        return (train, valid, test)

    @staticmethod
    def _create_vocab(datadir:str, dataset:str, vocab_type:str, vocab_cache_dir:str,
                      vocab_size:Optional[int]=None)->VocabBase:
        if vocab_type == 'word':
            # '<S>' is added for double eos and <unk> is rare token in corpus with freq < 3
            bos_token, eos_token, lower_case, vocab_file = None, '<eos>', False, None # vocab file is text file of symbols, one per line
            if dataset in ['wt103', 'wt2'] or dataset.startswith('olx_'):
                pass
            elif dataset == 'ptb':
                lower_case = True
            elif dataset == 'lm1b':
                bos_token, eos_token = '<S>', '<S>'
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

    def _dataset_filepaths(self)->Tuple[str,str,str]:
        train_filename, valid_filename, test_filename = 'train.txt', 'valid.txt', 'test.txt'
        if self.dataset in ['wt2', 'wt103']:
            train_filename, valid_filename, test_filename = 'wiki.train.tokens', 'wiki.valid.tokens', 'wiki.test.tokens'
        if self.dataset == 'lm1b':
            train_path_pattern = os.path.join(self.datadir, '1-billion-word-language-modeling-benchmark-r13output', 'training-monolingual.tokenized.shuffled', 'news.en-*')
            train_filename = glob.glob(train_path_pattern)
            return (train_filename, os.path.join(self.datadir, valid_filename), os.path.join(self.datadir, test_filename))

        return (os.path.join(self.datadir, train_filename),
                os.path.join(self.datadir, valid_filename),
                os.path.join(self.datadir, test_filename))

    def _train_vocab(self)->None:
        if self.refresh_cache or not self.vocab.is_trained(): # if vocab cache does not exist
            train_filepath, valid_filepath, test_filepath = \
                self._dataset_filepaths()

            logging.info('Training vocab...')
            if not isinstance(train_filepath, list):
                train_filepath = [train_filepath]
            self.vocab.train(train_filepath)
            logging.info('Finished training vocab.')
        else:
            self.vocab.load()
            logging.info(f'Vocab cache found and loaded for type {self.vocab_type} and size {self.vocab_size} from {self._vocab_cache_dir}.')

    def get_iterator(self, split, batch_size, tgt_len, device, ext_len, mem_len=None):
        if split == 'train':
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8'] or self.dataset.startswith('olx_'):
                data_iter = LMOrderedIterator(self.train, batch_size, tgt_len,
                                              device=device, ext_len=ext_len, mem_len=mem_len)
            elif self.dataset == 'lm1b':
                data_iter = LMMultiFileIterator(self.train, self.vocab, batch_size, tgt_len,
                                                device=device, ext_len=ext_len, mem_len=mem_len)
            else:
                raise RuntimeError(f'Dataset not yet fully supported: {self.dataset}')

        elif split in ['valid', 'test']:
            data = self.valid if split == 'valid' else self.test
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8'] or self.dataset.startswith('olx_'):
                data_iter = LMOrderedIterator(data, batch_size, tgt_len,
                                              device=device, ext_len=ext_len, mem_len=mem_len)
            elif self.dataset == 'lm1b':
                data_iter = LMShuffledIterator(data, batch_size, tgt_len,
                                               device=device, ext_len=ext_len, mem_len=mem_len)
            else:
                raise RuntimeError(f'Dataset not yet fully supported: {self.dataset}')

        else:
            raise RuntimeError(f'split not supported: {split}')

        return data_iter
