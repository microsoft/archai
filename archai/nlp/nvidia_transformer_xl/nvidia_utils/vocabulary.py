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

from archai.nlp.tokenizer_utils.token_dataset import TokenConfig, TokenizerFiles
import contextlib
import os
from collections import Counter
from collections import OrderedDict
from typing import Optional

import torch
from archai.nlp.nvidia_transformer_xl.nvidia_utils import distributed as nv_distributed

from archai.nlp.tokenizer_utils.token_trainer import create_tokenizer

class Vocab: # Word vocab is the default
    def __init__(self, special=[], min_freq=0, max_size=None, lower_case=True,
                 delimiter=None, vocab_file=None):
        """
        APIs:
            1. count_file -> count tokens
            2. build_vocab -> assign IDs to tokens
            3. encode_file -> convert file to IDs

        internal:
            tokenize -> split to tokens
            count_sents -> count freq from parsed sentenses

        """
        self.counter = Counter()
        self.special = special
        self.min_freq = min_freq
        self.max_size = max_size
        self.lower_case = lower_case
        self.delimiter = delimiter
        self.vocab_file = vocab_file # cached vocab file

    def tokenize(self, line, add_eos=False, add_double_eos=False):
        """Split line into tokens, add_eos: add special to end, add_double_eos: add special to begin and end"""
        line = line.strip()
        # convert to lower case
        if self.lower_case:
            line = line.lower()

        # empty delimiter '' will evaluate False
        if self.delimiter == '':
            symbols = line
        else:
            symbols = line.split(self.delimiter)

        if add_double_eos:  # lm1b
            return ['<S>'] + symbols + ['<S>']
        elif add_eos:
            return symbols + ['<eos>']
        else:
            return symbols

    def count_file(self, path, verbose=True, add_eos=False):
        """Setup counter with frequencies, return tokens for the entir file"""
        if verbose:
            print('counting file {} ...'.format(path))
        assert os.path.exists(path)

        # read lines, count frequencies of tokens, convert to tokens
        sents = [] # will contain all parsed tokens
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                symbols = self.tokenize(line, add_eos=add_eos)
                self.counter.update(symbols)
                sents.append(symbols)

        return sents

    def count_sents(self, sents, verbose=False):
        """
            sents : a list of sentences, each a list of tokenized symbols
        """
        if verbose:
            print('counting {} sents ...'.format(len(sents)))
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('    line {}'.format(idx))
            self.counter.update(symbols)

    def _load_from_file(self, vocab_file):
        """[This is not used] Load previously cached vocab file"""
        self.idx2sym = [] # clear out existing symbols
        self.sym2idx = OrderedDict()

        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                symb = line.strip().split()[0]
                self.add_symbol(symb)
        self.unk_idx = self.sym2idx['<unk>']

    def build_vocab(self):
        """Build the vocab by creating indices from the counter"""
        if self.vocab_file:
            print('Loading vocab from {}'.format(self.vocab_file))
            self._load_from_file(self.vocab_file)
            print('Final vocab size {}'.format(len(self)))
        else:
            print('Building vocab with min_freq={}, max_size={}'.format(
                self.min_freq, self.max_size))
            self.idx2sym = []
            self.sym2idx = OrderedDict()

            for sym in self.special:
                self.add_special(sym)

            for sym, cnt in self.counter.most_common(self.max_size):
                if cnt < self.min_freq:
                    break
                self.add_symbol(sym)

            print('final vocab size is {}, unique tokens are {}'.format(
                len(self), len(self.counter)))

    def encode_file(self, path, ordered=False, verbose=True, add_eos=True,
                    add_double_eos=False):
        if verbose:
            print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        encoded = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                symbols = self.tokenize(line, add_eos=add_eos,
                                        add_double_eos=add_double_eos)
                encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    def encode_sents(self, sents, ordered=False, verbose=True):
        if verbose:
            print('encoding {} sents ...'.format(len(sents)))
        encoded = []
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('    line {}'.format(idx))
            encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, '{}_idx'.format(sym.strip('<>')), self.sym2idx[sym])

    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def get_sym(self, idx):
        assert 0 <= idx < len(self), 'Index {} out of range'.format(idx)
        return self.idx2sym[idx]

    def get_idx(self, sym):
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        else:
            # print('encounter unk {}'.format(sym))
            assert '<eos>' not in sym
            assert hasattr(self, 'unk_idx')
            return self.sym2idx.get(sym, self.unk_idx)

    def get_symbols(self, indices):
        return [self.get_sym(idx) for idx in indices]

    def get_indices(self, symbols):
        return [self.get_idx(sym) for sym in symbols]

    def convert_to_tensor(self, symbols):
        return torch.LongTensor(self.get_indices(symbols))

    def convert_to_sent(self, indices, exclude=None):
        if exclude is None:
            return ' '.join([self.get_sym(idx) for idx in indices])
        else:
            return ' '.join([self.get_sym(idx) for idx in indices if idx not in exclude])

    def __len__(self):
        return len(self.idx2sym)


# Class OpenAIVocab has been adapted from
# https://github.com/cybertronai/transformer-xl/blob/master/utils/vocabulary.py
class OpenAIVocab(Vocab):
    def __init__(self, max_size=None, vocab_dir:Optional[str]=None):
        from pytorch_transformers import GPT2Tokenizer

        # GPT2Tokenizer
        # vocab_size: 50257
        # bos = eos = unk = '<|endoftext|>'
        # sep_token = None
        # max_model_input_sizes: {'gpt2': 1024, 'gpt2-medium': 1024, 'gpt2-large': 1024}
        # max_len = max_len_sentence_pair = max_len_single_sentence = 1024
        # mask_token = None

        if vocab_dir is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        else:
            tokenizer_files = TokenizerFiles.from_path(save_dir=vocab_dir)
            self.tokenizer = create_tokenizer(tokenizer_files, TokenConfig())

        self.EOT = self.tokenizer.encoder['<|endoftext|>']
        self.max_size = max_size

        pad = 8
        vocab_size = len(self.tokenizer)
        padded_vocab_size = (vocab_size + pad - 1) // pad * pad
        for i in range(0, padded_vocab_size - vocab_size):
            token = f'madeupword{i:09d}'
            self.tokenizer.add_tokens([token])

    def __len__(self):
        return len(self.tokenizer)

    def count_file(self, path, verbose=False, add_eos=False):
        raise RuntimeError('count_file should not be called in OpenAIVocab')

    def build_vocab(self):
        pass

    def encode_file(self, path, ordered=False, verbose=False, add_eos=True, add_double_eos=False) -> torch.LongTensor:
        # Suppress warnings about length.
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            out = torch.LongTensor(self.tokenizer.encode(f.read()) + [self.EOT])
            return out

    def tokenize(self, line, add_eos=False, add_double_eos=False):
        return self.tokenizer.encode(line)

    def convert_to_tensor(self, symbols):
        return torch.LongTensor(symbols)


