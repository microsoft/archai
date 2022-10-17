import os
import logging
from collections import Counter
from collections import OrderedDict
from typing import List, Optional
import pathlib

import torch

from overrides import overrides

from archai.nlp.datasets.tokenizer_utils.vocab_base import VocabBase
from archai.common import utils
from archai.nlp.datasets import distributed_utils
from archai.nlp.datasets.tokenizer_utils.token_config import TokenConfig
from archai.nlp.datasets.tokenizer_utils.special_token_enum import SpecialTokenEnum

class WordVocab(VocabBase): # Word vocab is the default
    def __init__(self, save_path:str, min_freq=0, vocab_size=None,
                 bos_token:Optional[str]=None, eos_token:Optional[str]='<eos>',
                 unk_token:Optional[str]='<unk>', lower_case=False,
                 delimiter=None, encode_special_tokens=True, decode_special_tokens=True):
        self.counter = Counter()
        self._config = TokenConfig(bos_token=bos_token, eos_token=eos_token,
                                   unk_token=unk_token, pad_token=None,
                                   # no prefix space or line needed as we delimit on white space unlike in bbpe
                                   add_prefix_space=False, add_prefix_new_line=False,
                                   lower_case=lower_case)

        assert self._config.unk_token, "unk token must be supplied for WordVocab"
        self._bos = [self._config.bos_token] if self._config.bos_token else []
        self._eos = [self._config.eos_token] if self._config.eos_token else []

        self.encode_special_tokens = encode_special_tokens
        self.decode_special_tokens = decode_special_tokens
        self.min_freq = min_freq
        self.vocab_size = vocab_size
        self.delimiter = delimiter
        self.save_path = save_path

    # TODO: remove suplicates of this function from across the project
    def _preprocess_text(self, text:str)->str:
        #text = text.strip()
        if self._config.add_prefix_space:
            text = ' ' + text
        if self._config.add_prefix_new_line:
            text = '\n' + text
        if self._config.lower_case:
            text = text.lower()
        return text

    def _add_file(self, path, verbose=True)->None:
        """Setup counter with frequencies, return tokens for the entir file"""
        if verbose:
            logging.info('counting file {} ...'.format(path))
        assert os.path.exists(path), f"Training file to build word vocan does not exist: {path}"

        # read lines, count frequencies of tokens, convert to tokens
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    logging.info('    file line {}'.format(idx))
                symbols = self._tokenize_text(line)
                self.counter.update(symbols)

    def _tokenize_text(self, text:str)->List[str]:
        text = self._preprocess_text(text)

        # split on whitespace
        symbols = text.split(self.delimiter)

        return symbols

    def _count_sents(self, sents, verbose=False):
        """
            sents : a list of sentences, each a list of tokenized symbols
        """
        if verbose:
            logging.info('counting {} sents ...'.format(len(sents)))
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                logging.info('    file line {}'.format(idx))
            self.counter.update(symbols)

    def _clear(self):
        self.idx2sym:List[str] = [] # clear out existing symbols
        self.sym2idx:OrderedDict[str, int] = OrderedDict()

    @overrides
    def load(self)->None:
        """Load previously cached vocab file"""
        vocab_filepath = self._vocab_filepath()

        self._clear() # clear out existing symbols

        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                symb = line.strip().split()[0]
                self._add_symbol(symb)
        self.unk_idx = self.sym2idx[self._config.unk_token]

    def _vocab_filepath(self)->str:
        return utils.full_path(os.path.join(self.save_path, 'vocab.txt'))

    def _save(self, path:str)->None:
        path = utils.full_path(self.save_path, create=True)
        vocab_filepath = self._vocab_filepath()
        with open(vocab_filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.idx2sym))

    @overrides
    def is_trained(self)->bool:
        vocab_filepath = self._vocab_filepath()
        return os.path.exists(vocab_filepath)

    @overrides
    def train(self, filepaths:List[str])->None:
        logging.info(f'Building word vocab with min_freq={self.min_freq}, vocab_size={self.vocab_size} using {len(filepaths)} training file(s) at {self.save_path}')

        assert len(filepaths)

        self._clear()

        for filepath in filepaths:
            self._add_file(filepath)

        # add specials regardless of vocab_size
        for sym in self._config.get_special_tokens():
            self._add_special(sym)

        remaining_len = self.vocab_size - len(self) if self.vocab_size is not None else None
        for sym, cnt in self.counter.most_common(remaining_len):
            if cnt < self.min_freq:
                break # don't add rare words
            self._add_symbol(sym)

        with distributed_utils.distributed.sync_workers() as rank:
            if rank == 0:
                self._save(self.save_path)

        logging.info(f'Final word vocab size is {len(self)}, unique tokens are {len(self.counter)}')

    @overrides
    def encode_text(self, text:str)->List[int]:
        symbols = self._tokenize_text(text)

        if self.encode_special_tokens:
            symbols = self._bos + symbols + self._eos

        toks = self._get_indices(symbols)

        return toks

    @overrides
    def decode_text(self, ids:List[int])->str:
        syms = self.ids_to_tokens(ids)
        if self.decode_special_tokens and len(syms):
            if syms[0] == self._bos:
                syms = syms[1:]
            if len(syms) and syms[-1] == self._eos:
                syms = syms[:-1]
        return ' '.join(syms)

    @overrides
    def special_token_id(self, sp:SpecialTokenEnum)->Optional[int]:
        return self.token_to_id(self._config.special_token_name(sp))

    def _encode_sents(self, sents, ordered=False, verbose=True):
        if verbose:
            logging.info('encoding {} sents ...'.format(len(sents)))
        encoded = []
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                logging.info('    line {}'.format(idx))
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
            return self.sym2idx.get(sym, self.unk_idx)

    def _indices2symbols(self, indices)->List[str]:
        return [self._get_sym(idx) for idx in indices]

    @overrides
    def token_to_id(self, t:str)->int:
        return self._get_idx(t)
    @overrides
    def id_to_token(self, id:int)->str:
        return self._get_sym(id)
    @overrides
    def tokens_to_ids(self, ts:List[str])->List[int]:
        return self._get_indices(ts)
    @overrides
    def ids_to_tokens(self, ids:List[int])->List[str]:
        return self._indices2symbols(ids)

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
