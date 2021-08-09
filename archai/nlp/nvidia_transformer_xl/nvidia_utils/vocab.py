import os
from collections import Counter
from collections import OrderedDict
from typing import List, Optional

import torch


class Vocab: # Word vocab is the default
    def __init__(self, special=[], min_freq=0, max_size=None, lower_case=True,
                 delimiter=None, vocab_file=None):
        """
        APIs:
            1. add_file -> count tokens
            2. build_vocab -> assign IDs to tokens
            3. tokenize_file -> convert file to IDs

        internal:
            _get_symbols -> split to symbols
            _count_sents -> count freq from parsed sentenses

        """
        self.counter = Counter()
        self.special = special
        self.min_freq = min_freq
        self.max_size = max_size
        self.lower_case = lower_case
        self.delimiter = delimiter
        self.vocab_file = vocab_file # cached vocab file

    def _get_symbols(self, line, add_eos=False, add_double_eos=False)->List[str]:
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

        if add_double_eos:  # lm1b
            return ['<S>'] + symbols + ['<S>']
        elif add_eos:
            return symbols + ['<eos>']
        else:
            return symbols

    def add_file(self, path, verbose=True, add_eos=False)->None:
        """Setup counter with frequencies, return tokens for the entir file"""
        if verbose:
            print('counting file {} ...'.format(path))
        assert os.path.exists(path)

        # read lines, count frequencies of tokens, convert to tokens
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                symbols = self._get_symbols(line, add_eos=add_eos)
                self.counter.update(symbols)

    def _count_sents(self, sents, verbose=False):
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
                self._add_symbol(symb)
        self.unk_idx = self.sym2idx['<unk>']

    def build_vocab(self):
        """Build the vocab by creating indices from the current counter"""
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
                self._add_special(sym)

            for sym, cnt in self.counter.most_common(self.max_size):
                if cnt < self.min_freq:
                    break
                self._add_symbol(sym)

            print('final vocab size is {}, unique tokens are {}'.format(
                len(self), len(self.counter)))

    def tokenize_file(self, path, ordered=False, verbose=True, add_eos=True,
                    add_double_eos=False):
        if verbose:
            print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        encoded = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                tokens = self.tokenize_line(line, add_eos=add_eos,
                                        add_double_eos=add_double_eos)
                encoded.append(tokens)

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    def tokenize_line(self, line, add_eos=True, add_double_eos=False):
        symbols = self._get_symbols(line, add_eos=add_eos, add_double_eos=add_double_eos)
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

    def _get_idx(self, sym):
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        else:
            # print('encounter unk {}'.format(sym))
            assert '<eos>' not in sym
            assert hasattr(self, 'unk_idx')
            return self.sym2idx.get(sym, self.unk_idx)

    def _inices2symbols(self, indices):
        return [self._get_sym(idx) for idx in indices]

    def _get_indices(self, symbols):
        return [self._get_idx(sym) for sym in symbols]

    def _convert_to_tensor(self, symbols):
        return torch.LongTensor(self._get_indices(symbols))

    def _convert_to_sent(self, indices, exclude=None):
        if exclude is None:
            return ' '.join([self._get_sym(idx) for idx in indices])
        else:
            return ' '.join([self._get_sym(idx) for idx in indices if idx not in exclude])

    def __len__(self):
        return len(self.idx2sym)

