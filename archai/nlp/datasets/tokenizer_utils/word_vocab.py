# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Word-based vocabulary.
"""

import logging
import os
from collections import Counter, OrderedDict
from typing import List, Optional, Union

import torch
from archai.common import utils
from archai.nlp.datasets.distributed_utils import distributed
from archai.nlp.datasets.tokenizer_utils.special_token_enum import \
    SpecialTokenEnum
from archai.nlp.datasets.tokenizer_utils.token_config import TokenConfig
from archai.nlp.datasets.tokenizer_utils.vocab_base import VocabBase
from overrides import overrides


class WordVocab(VocabBase): # Word vocab is the default
    """Word-based vocabulary.

    """

    def __init__(self,
                 save_path: str,
                 min_freq: Optional[int] = 0,
                 vocab_size: Optional[int] = None,
                 bos_token: Optional[str] = None,
                 eos_token: Optional[str] = '<eos>',
                 unk_token: Optional[str] = '<unk>',
                 lower_case: Optional[bool] = False,
                 delimiter: Optional[str] = None,
                 encode_special_tokens: Optional[bool] = True,
                 decode_special_tokens: Optional[bool] = True) -> None:
        """Overrides initialization method.

        Args:
            save_path: Output path.
            min_freq: Minimum frequency of tokens.
            vocab_size: Vocabulary size.
            bos_token: Begin-of-sentence token.
            eos_token: End-of-sentence token.
            unk_token: Unknown token.
            lower_case: Whether to apply lower case or not.
            delimiter: Delimiter used to split the tokens.
            encode_special_tokens: Whether to decode special tokens or not.
            decode_special_tokens: Whether to decode special tokens or not.

        """

        self.counter = Counter()
        self._config = TokenConfig(bos_token=bos_token, eos_token=eos_token,
                                   unk_token=unk_token, pad_token=None,
                                   # no prefix space or line needed as we delimit on white space unlike in bbpe
                                   add_prefix_space=False, add_prefix_new_line=False,
                                   lower_case=lower_case)

        assert self._config.unk_token, 'unk token must be supplied for WordVocab'
        self._bos = [self._config.bos_token] if self._config.bos_token else []
        self._eos = [self._config.eos_token] if self._config.eos_token else []

        self.encode_special_tokens = encode_special_tokens
        self.decode_special_tokens = decode_special_tokens
        self.min_freq = min_freq
        self.vocab_size = vocab_size
        self.delimiter = delimiter
        self.save_path = save_path

    # TODO: remove suplicates of this function from across the project
    def _preprocess_text(self, text: str) -> str:
        """Pre-process the input text.

        Args:
            text: Text to be pre-processed.

        Returns:
            (str): Pre-processed text.

        """

        #text = text.strip()
        if self._config.add_prefix_space:
            text = ' ' + text
        if self._config.add_prefix_new_line:
            text = '\n' + text
        if self._config.lower_case:
            text = text.lower()

        return text

    def _add_file(self, path: str, verbose: Optional[bool] = True) -> None:
        """Setups the counter with frequencies and convert to tokens.
        
        Args:
            path: Input file path.
            verbose: Whether to display additional logging or not.
            
        """

        if verbose:
            logging.info('counting file {} ...'.format(path))

        assert os.path.exists(path), f"Training file to build word vocab does not exist: {path}"

        # read lines, count frequencies of tokens, convert to tokens
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    logging.info('    file line {}'.format(idx))

                symbols = self._tokenize_text(line)
                self.counter.update(symbols)

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenizes the text bt applying pre-procesing and delimiter splitting.

        Args:
            text: Input text.
        
        Returns:
            (List[str]): Tokenized text.

        """

        text = self._preprocess_text(text)

        # split on whitespace
        symbols = text.split(self.delimiter)

        return symbols

    def _count_sents(self,
                     sents: List[str],
                     verbose: Optional[bool] = False) -> None:
        """Counts the sentences and updates the tokens' counter.

        Args:
            sents: List of sentences, where each list is composed of tokenized symbols.
            verbose: Whether to display additional logging or not.

        """

        if verbose:
            logging.info('counting {} sents ...'.format(len(sents)))

        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                logging.info('    file line {}'.format(idx))

            self.counter.update(symbols)

    def _clear(self) -> None:
        """Clears the existing symbols (vocabulary).

        """

        self.idx2sym:List[str] = [] # clear out existing symbols
        self.sym2idx:OrderedDict[str, int] = OrderedDict()

    @overrides
    def load(self) -> None:
        """Loads a pre-trained vocabulary.

        """

        vocab_filepath = self._vocab_filepath()

        self._clear() # clear out existing symbols

        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                symb = line.strip().split()[0]
                self._add_symbol(symb)

        self.unk_idx = self.sym2idx[self._config.unk_token]

    def _vocab_filepath(self) -> str:
        """Gathers the vocabulary file path.

        Returns:
            (str): Vocabulary file path.
        """

        return utils.full_path(os.path.join(self.save_path, 'vocab.txt'))

    def _save(self, path: str) -> None:
        """Saves the vocabulary to a pre-defined path.

        Args:
            path: Path to be saved.

        """

        path = utils.full_path(self.save_path, create=True)
        vocab_filepath = self._vocab_filepath()

        with open(vocab_filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.idx2sym))

    @overrides
    def is_trained(self) -> bool:
        """Checks whether vocabulary has already been trained.

        Returns:
            (bool): Whether vocabulary has already been trained.

        """

        vocab_filepath = self._vocab_filepath()

        return os.path.exists(vocab_filepath)

    @overrides
    def train(self, filepaths: List[str]) -> None:
        """Trains a new vocabulary.

        Args:
            filepaths: Path to the input files.

        """

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

        with distributed.sync_workers() as rank:
            if rank == 0:
                self._save(self.save_path)

        logging.info(f'Final word vocab size is {len(self)}, unique tokens are {len(self.counter)}')

    @overrides
    def encode_text(self, text: str) -> List[int]:
        """Encodes input text into a list of tokens.

        Args:
            text: Input text.

        Returns:
            (List[int]): Encoded tokens.

        """

        symbols = self._tokenize_text(text)

        if self.encode_special_tokens:
            symbols = self._bos + symbols + self._eos

        toks = self._get_indices(symbols)

        return toks

    @overrides
    def decode_text(self, ids: List[int]) -> str:
        """Decodes tokens into a string-based text.

        Args:
            ids: Tokens identifiers.

        Returns:
            (str): Decoded tokens.

        """

        syms = self.ids_to_tokens(ids)

        if self.decode_special_tokens and len(syms):
            if syms[0] == self._bos:
                syms = syms[1:]

            if len(syms) and syms[-1] == self._eos:
                syms = syms[:-1]

        return ' '.join(syms)

    @overrides
    def special_token_id(self, sp: SpecialTokenEnum) -> int:
        """Gets the special token identifier.

        Args:
            sp: Special token enumerator.

        Returns:
            (int): Special token identifier.

        """

        return self.token_to_id(self._config.special_token_name(sp))

    def _encode_sents(self,
                      sents: List[str],
                      ordered: Optional[bool] = False,
                      verbose: Optional[bool] = True) -> Union[List[List[int]], torch.Tensor]:
        """Encodes a list of sentences into a tensor.
        
        Args:
            sents: List of sentences.
            ordered: Whether to concat sentences into a single tensor or not.
            verbose: Whether to apply additional logging or not.
        
        Returns:
            (Union[List[List[int]], torch.Tensor]): Encoded sentences.

        """

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

    def _add_special(self, sym: str) -> None:
        """Adds special token to the vocabulary.

        Args:
            sym: Special token to be added.

        """

        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

            setattr(self, '{}_idx'.format(sym.strip('<>')), self.sym2idx[sym])

    def _add_symbol(self, sym: str) -> None:
        """Adds symbols to the vocabulary.

        Args:
            sym: Symbol to be added.

        """

        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def _get_sym(self, idx: int) -> str:
        """Gets corresponding symbol based on supplied identifier.

        Args:
            idx: Identifier.
        
        Returns:
            (str): Symbol.

        """

        assert 0 <= idx < len(self), 'Index {} out of range'.format(idx)

        return self.idx2sym[idx]

    def _get_idx(self, sym: str) -> int:
        """Gets identifier based on supplied symbol.

        Args:
            sym: Symbol.

        Returns:
            (int): Symbol identifier.

        """

        if sym in self.sym2idx:
            return self.sym2idx[sym]
        else:
            return self.sym2idx.get(sym, self.unk_idx)

    def _indices2symbols(self, indices: List[int]) -> List[str]:
        """Converts a list of indices to a list of symbols.

        Args:
            indices: List of indices.

        Returns:
            (List[str]): List of symbols.

        """

        return [self._get_sym(idx) for idx in indices]

    @overrides
    def token_to_id(self, t: str) -> int:
        """Converts a string-based token to an identifier.

        Args:
            t: String-based token.

        Returns:
            (int): Token identifier.

        """

        return self._get_idx(t)

    @overrides
    def id_to_token(self, id: int) -> str:
        """Converts a token identifier to a string-based token.

        Args:
            id: Token identifier.

        Returns:
            (str): String-based token.

        """

        return self._get_sym(id)

    @overrides
    def tokens_to_ids(self, ts: List[str]) -> List[int]:
        """Converts a list of string-based tokens to their identifiers.

        Args:
            ts: List of string-based tokens.

        Returns:
            (List[int]): List of tokens identifiers.

        """

        return self._get_indices(ts)

    @overrides
    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Converts a list of tokens identifiers to their string-based versions.

        Args:
            ids: List of tokens identifiers.

        Returns:
            (List[int]): List of string-based tokens.

        """

        return self._indices2symbols(ids)

    def _get_indices(self, symbols: List[str]) -> List[int]:
        """Gets a list of indices based on supplied symbols.
        
        Args:
            symbols: List of symbols.

        Returns:
            (List[int]): List of symbols identifiers.

        """

        return [self._get_idx(sym) for sym in symbols]

    def _convert_to_tensor(self, symbols: List[str]) -> torch.LongTensor:
        """Converts a list of indices to a tensor.
        
        Args:
            symbols: List of symbols.

        Returns:
            (torch.LongTensor): Tensor of symbols identifiers.

        """

        return torch.LongTensor(self._get_indices(symbols))

    def _convert_to_sent(self,
                         indices: List[int],
                         exclude: Optional[str] = None) -> List[str]:
        """Converts a list of identifiers to a sentence.

        Args:
            indices: List of identifiers.
            exclude: Identifier to be excluded.

        Returns:
            (List[str]): Converted sentence.
            
        """

        if exclude is None:
            return ' '.join([self._get_sym(idx) for idx in indices])
        else:
            return ' '.join([self._get_sym(idx) for idx in indices if idx not in exclude])

    @overrides
    def __len__(self) -> int:
        """Calculates the size of the vocabulary.

        Args:
            (int): Size of vocabulary.

        """

        return len(self.idx2sym)
