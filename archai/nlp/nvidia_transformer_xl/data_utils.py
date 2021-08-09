# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import logging
import os
import re

import numpy as np
import sacremoses
import torch

from archai.nlp.nvidia_transformer_xl import nvidia_utils as utils
from archai.nlp.nvidia_transformer_xl.nvidia_utils.gpt_vocab import GptVocab
from archai.nlp.nvidia_transformer_xl.nvidia_utils.vocabulary import Vocab
from archai.common import utils

class LMOrderedIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', mem_len=None, ext_len=None, warmup=True):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0
        self.mem_len = mem_len
        self.warmup = warmup

        self.device = device

        # Work out how cleanly we can divide the dataset into bsz parts.
        n_step = data.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data[:n_step * bsz]

        # Evenly divide the data across the bsz batches.
        self.data = data.view(bsz, -1).t().contiguous().pin_memory()

        if mem_len and warmup:
            self.warmup_batches = (mem_len + bptt - 1) // bptt
            self.warmup_elems = self.warmup_batches * bptt

            warmup_data = self.data.roll((self.warmup_elems, 1), (0, 1))[:self.warmup_elems]
            self.data = torch.cat((warmup_data, self.data))

        # Partition data for DistributedDataParallel
        world_size = utils.distributed.get_world_size()
        rank = utils.distributed.get_rank()
        self.data = self.data.chunk(world_size, dim=1)[rank]

        # Number of mini-batches
        self.n_batch = (self.data.size(0) + self.bptt - 1) // self.bptt

        self.last_iter = None

    def roll(self, seed):
        rng = torch.Generator()
        rng.manual_seed(seed)
        for i in range(self.data.size(1)):
            row = self.data[:, i]
            shift = torch.randint(0, self.data.size(0), (1,), generator=rng)
            row = torch.cat((row[shift:], row[:shift]))
            self.data[:, i] = row

    def get_batch(self, i, bptt=None):
        if bptt is None:
            bptt = self.bptt

        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx].to(self.device, non_blocking=True)
        target = self.data[i+1:i+1+seq_len].to(self.device, non_blocking=True)

        if self.mem_len and self.warmup:
            warm = i >= self.warmup_elems
        else:
            warm = True

        return data, target, seq_len, warm

    def get_fixlen_iter(self, start=0):
        if start != 0:
            start += self.bptt
        for i in range(start, self.data.size(0) - 1, self.bptt):
            self.last_iter = i
            yield self.get_batch(i)

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len = self.get_batch(i, bptt)
            i += seq_len
            yield data, target, seq_len
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        return self.get_fixlen_iter()


class LMShuffledIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None, shuffle=False):
        """
            data -- list[LongTensor] -- there is no order among the LongTensors
        """
        self.data = data

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self):
        # index iterator
        epoch_indices = np.random.permutation(len(self.data)) if self.shuffle \
            else np.array(range(len(self.data)))

        # sentence iterator
        for idx in epoch_indices:
            yield self.data[idx]

    def stream_iterator(self, sent_stream):
        # streams for each data in the batch
        streams = [None] * self.bsz

        data = torch.LongTensor(self.bptt, self.bsz)
        target = torch.LongTensor(self.bptt, self.bsz)

        n_retain = 0

        while True:
            # data   : [n_retain+bptt x bsz]
            # target : [bptt x bsz]
            data[n_retain:].fill_(-1)
            target.fill_(-1)

            valid_batch = True

            for i in range(self.bsz):
                n_filled = 0
                try:
                    while n_filled < self.bptt:
                        if streams[i] is None or len(streams[i]) <= 1:
                            streams[i] = next(sent_stream)
                        # number of new tokens to fill in
                        n_new = min(len(streams[i]) - 1, self.bptt - n_filled)
                        # first n_retain tokens are retained from last batch
                        data[n_retain+n_filled:n_retain+n_filled+n_new, i] = \
                            streams[i][:n_new]
                        target[n_filled:n_filled+n_new, i] = \
                            streams[i][1:n_new+1]
                        streams[i] = streams[i][n_new:]
                        n_filled += n_new
                except StopIteration:
                    valid_batch = False
                    break

            if not valid_batch:
                return

            data = data.to(self.device)
            target = target.to(self.device)

            yield data, target, self.bptt

            n_retain = min(data.size(0), self.ext_len)
            if n_retain > 0:
                data[:n_retain] = data[-n_retain:]
            data.resize_(n_retain + self.bptt, data.size(1))

    def __iter__(self):
        # sent_stream is an iterator
        sent_stream = self.get_sent_stream()

        for batch in self.stream_iterator(sent_stream):
            yield batch


class LMMultiFileIterator(LMShuffledIterator):
    def __init__(self, paths, vocab, bsz, bptt, device='cpu', ext_len=None,
                 shuffle=False):

        self.paths = paths
        self.vocab = vocab

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self, path):
        sents = self.vocab.tokenize_file(path, add_double_eos=True)
        if self.shuffle:
            np.random.shuffle(sents)
        sent_stream = iter(sents)

        return sent_stream

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.paths)

        for path in self.paths:
            # sent_stream is an iterator
            sent_stream = self.get_sent_stream(path)
            for batch in self.stream_iterator(sent_stream):
                yield batch


class Corpus(object):
    def __init__(self, datadir, dataset, vocab, max_size=None): # by default no args and kwargs are passed
        self.dataset = dataset
        if vocab == 'word':
            special, lower_case, vocab_file = [], True, None
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

            special += ['<S>', '<unk>'] # '<S>' is added for dounle eos and <unk> is rare token in corpus with freq < 3
            self.vocab = Vocab(max_size=max_size,
                               special=special,
                               lower_case=lower_case,
                               vocab_file=vocab_file)
        elif vocab == 'bpe':
            vocab_dir = utils.full_path(os.path.join(datadir, 'wikitext-103-bpe-vocab', str(max_size)), create=True)
            self.vocab = GptVocab(max_size=max_size or 50000, vocab_dir=vocab_dir)
        else:
            raise RuntimeError('Unsupported vocab')

        train_filename, test_filename, valid_filename = 'train.txt', 'test.txt', 'valid.txt'
        if self.dataset in ['wt2', 'wt103']:
            train_filename, test_filename, valid_filename = 'wiki.train.tokens', 'wiki.test.tokens', 'wiki.valid.tokens'

        if self.dataset in ['ptb', 'wt2', 'enwik8', 'text8']:
            self.vocab.add_file(os.path.join(datadir, train_filename))
            self.vocab.add_file(os.path.join(datadir, valid_filename))
            self.vocab.add_file(os.path.join(datadir, test_filename))
        elif self.dataset == 'wt103':
            self.vocab.add_file(os.path.join(datadir, train_filename))
        elif self.dataset == 'lm1b':
            train_path_pattern = os.path.join(
                datadir, '1-billion-word-language-modeling-benchmark-r13output',
                'training-monolingual.tokenized.shuffled', 'news.en-*')
            train_paths = glob.glob(train_path_pattern)
            # the vocab will load from file when build_vocab() is called

        self.vocab.build_vocab()

        if self.dataset in ['ptb', 'wt2', 'wt103']:
            self.train = self.vocab.tokenize_file(
                os.path.join(datadir, train_filename), ordered=True)
            self.valid = self.vocab.tokenize_file(
                os.path.join(datadir, valid_filename), ordered=True)
            self.test = self.vocab.tokenize_file(
                os.path.join(datadir, test_filename), ordered=True)
        elif self.dataset in ['enwik8', 'text8']:
            self.train = self.vocab.tokenize_file(
                os.path.join(datadir, train_filename), ordered=True, add_eos=False)
            self.valid = self.vocab.tokenize_file(
                os.path.join(datadir, valid_filename), ordered=True, add_eos=False)
            self.test = self.vocab.tokenize_file(
                os.path.join(datadir, test_filename), ordered=True, add_eos=False)
        elif self.dataset == 'lm1b':
            self.train = train_paths
            self.valid = self.vocab.tokenize_file(
                os.path.join(datadir, valid_filename), ordered=False, add_double_eos=True)
            self.test = self.vocab.tokenize_file(
                os.path.join(datadir, test_filename), ordered=False, add_double_eos=True)

    def get_iterator(self, split, *args, **kwargs):
        if split == 'train':
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
                data_iter = LMOrderedIterator(self.train, *args, **kwargs)
            elif self.dataset == 'lm1b':
                kwargs['shuffle'] = True
                data_iter = LMMultiFileIterator(self.train, self.vocab, *args, **kwargs)
        elif split in ['valid', 'test']:
            data = self.valid if split == 'valid' else self.test
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
                data_iter = LMOrderedIterator(data, *args, **kwargs)
            elif self.dataset == 'lm1b':
                data_iter = LMShuffledIterator(data, *args, **kwargs)

        return data_iter


def get_lm_corpus(datadir, cachedir, dataset, vocab, max_size=None):
    if vocab == 'word':
        fn = os.path.join(cachedir, 'cache.' + str(max_size) + '.word.v1.pt')
    elif vocab == 'bpe':
        fn = os.path.join(cachedir, 'cache.' + str(max_size) + '.bpe.v1.pt')
    else:
        raise RuntimeError('Unsupported vocab')

    if os.path.exists(fn):
        logging.info('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        logging.info('Producing dataset {}...'.format(dataset))
        corpus = Corpus(datadir, dataset, vocab, max_size=max_size)
        with utils.distributed.sync_workers() as rank:
            if rank == 0:
                torch.save(corpus, fn)

    return corpus


def tokenize_raw(text, lang='en'):
    mt = sacremoses.MosesTokenizer(lang)
    text = mt.tokenize(text, return_str=True)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&apos;', "'", text)
    text = re.sub(r'(\d)\.(\d)', r'\1 @.@ \2', text)
    text = re.sub(r'(\d),(\d)', r'\1 @,@ \2', text)
    text = re.sub(r'(\w)-(\w)', r'\1 @-@ \2', text)
    return text


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='unit test')
    parser.add_argument('--datadir', type=str, default='../data/text8',
                        help='location of the data corpus')
    parser.add_argument('--cachedir', type=str, default='~/logdir/data/text8',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='text8',
                        choices=['ptb', 'wt2', 'wt103', 'lm1b', 'enwik8', 'text8'],
                        help='dataset name')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    corpus = get_lm_corpus(args.datadir, args.cachedir, args.dataset, vocab='word')
    logging.info('Vocab size : {}'.format(len(corpus.vocab.idx2sym)))
