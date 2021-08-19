import os
from collections import Counter
from collections import OrderedDict
from typing import List, Optional
import pathlib

import torch

from overrides import overrides

from archai.nlp.nvidia_transformer_xl.nvidia_utils.vocab_base import VocabBase
from archai.common import utils
from archai.nlp.nvidia_transformer_xl import nvidia_utils as nv_utils

class WordVocab(VocabBase): # Word vocab is the default
    def __init__(self, special=[], min_freq=0, vocab_size=None, lower_case=True,
                 delimiter=None, add_eos=False, add_double_eos=False):
        self.counter = Counter()
        self.special = special
        self.min_freq = min_freq
        self.vocab_size = vocab_size
        self.lower_case = lower_case
        self.delimiter = delimiter
        self.add_eos = add_eos
        self.add_double_eos = add_double_eos

    def _tokenize_line(self, line)->List[str]:
        """Tokenize line, split on space, add_eos: add special to end, add_double_eos: add special to begin and end"""
        line = line.strip()
        # convert to lower case
        if self.lower_case:
            line = line.lower()

        # empty delimiter '' will evaluate False
        if self.delimiter == '':
            symbols = line
        else:
            symbols = line.split(self.delimiter)

        if self.add_double_eos:  # lm1b
            return ['<S>'] + symbols + ['<S>']
        elif self.add_eos:
            return symbols + ['<eos>']
        else:
            return symbols

    def _add_file(self, path, verbose=True)->None:
        """Setup counter with frequencies, return tokens for the entir file"""
        if verbose:
            print('counting file {} ...'.format(path))
        assert os.path.exists(path)

        # read lines, count frequencies of tokens, convert to tokens
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    file line {}'.format(idx))
                symbols = self._tokenize_line(line)
                self.counter.update(symbols)

    def _count_sents(self, sents, verbose=False):
        """
            sents : a list of sentences, each a list of tokenized symbols
        """
        if verbose:
            print('counting {} sents ...'.format(len(sents)))
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('    file line {}'.format(idx))
            self.counter.update(symbols)

    def _erase(self):
        self.idx2sym:List[str] = [] # clear out existing symbols
        self.sym2idx:OrderedDict[str, int] = OrderedDict()

    @overrides
    def load(self, path:str)->bool:
        """Load previously cached vocab file"""

        cach_filepath = utils.full_path(os.path.join(path, 'vocab.txt'))
        if os.path.exists(cach_filepath):
            self._erase() # clear out existing symbols

            with open(cach_filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    symb = line.strip().split()[0]
                    self._add_symbol(symb)
            self.unk_idx = self.sym2idx['<unk>']
            return True
        else:
            return False

    def _save(self, path:str)->None:
        path = utils.full_path(path, create=True)
        cach_filepath = os.path.join(path, 'vocab.txt')
        with open(cach_filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.idx2sym))

    @overrides
    def train(self, filepaths:List[str], save_dir:str)->None:
        print('Building word vocab with min_freq={}, vocab_size={}'.format(
            self.min_freq, self.vocab_size))

        self._erase()

        for filepath in filepaths:
            self._add_file(filepath)

        for sym in self.special:
            self._add_special(sym)

        for sym, cnt in self.counter.most_common(self.vocab_size):
            if cnt < self.min_freq:
                break
            self._add_symbol(sym)

        with nv_utils.distributed.sync_workers() as rank:
            if rank == 0:
                self._save(save_dir)

        print('final vocab size is {}, unique tokens are {}'.format(
            len(self), len(self.counter)))

    @overrides
    def encode_line(self, line):
        symbols = self._tokenize_line(line)
        return self._convert_to_tensor(symbols)

    def _encode_sents(self, sents, ordered=False, verbose=True):
        if verbose:
            print('encoding {} sents ...'.format(len(sents)))
        encoded = []
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('    line {}'.format(idx))
            encoded.append(self._convert_to_tensor(symbols))

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    def _add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, '{}_idx'.format(sym.strip('<>')), self.sym2idx[sym])

    def _add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def _get_sym(self, idx):
        assert 0 <= idx < len(self), 'Index {} out of range'.format(idx)
        return self.idx2sym[idx]

    def _get_idx(self, sym)->int:
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        else:
            # print('encounter unk {}'.format(sym))
            assert '<eos>' not in sym
            assert hasattr(self, 'unk_idx')
            return self.sym2idx.get(sym, self.unk_idx)

    def _indices2symbols(self, indices)->List[str]:
        return [self._get_sym(idx) for idx in indices]

    def _get_indices(self, symbols)->List[int]:
        return [self._get_idx(sym) for sym in symbols]

    def _convert_to_tensor(self, symbols)->torch.LongTensor:
        return torch.LongTensor(self._get_indices(symbols))

    def _convert_to_sent(self, indices, exclude=None):
        if exclude is None:
            return ' '.join([self._get_sym(idx) for idx in indices])
        else:
            return ' '.join([self._get_sym(idx) for idx in indices if idx not in exclude])

    @overrides
    def __len__(self):
        return len(self.idx2sym)

