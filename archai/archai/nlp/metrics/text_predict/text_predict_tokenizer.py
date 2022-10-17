# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tokenizer-based class that works with the Text Predictor.
"""

import functools
import re
from typing import List, Optional, Set, Tuple

import numpy as np
from transformers import PreTrainedTokenizerFast

from archai.common.lru_cache import LRUCache

# Token-related constants
TOKENIZER_FILTER_TOKEN_IDS_CACHE_SIZE = 65536
TOKENIZER_WORD_TOKEN_SEPARATOR = 'Ġ \nĊ\t\.;:,\'\"`<>\(\)\{\}\[\]\|\!@\#\$\%\^\&\*=\+\?/\\_\-~'
TOKENIZER_WORD_TOKEN_SEPARATOR_SET = set(TOKENIZER_WORD_TOKEN_SEPARATOR)

# Regex
REGEX_SPLIT = re.compile('^(.*)([' + TOKENIZER_WORD_TOKEN_SEPARATOR + '].*)$', re.MULTILINE | re.DOTALL)
REGEX_WHITESPACE = re.compile("[\xa0 \t\u2002-\u2006\u200B]+", re.MULTILINE | re.DOTALL)
REGEX_NEW_LINE = re.compile("\s*[\r\n]+\s*", re.MULTILINE | re.DOTALL)


class TextPredictTokenizer:
    """Loads an pre-trained tokenizer (.json) and complies with Text Predict.

    """

    # Set of tokens that should not be shown to the user
    INVALID_TOKENS = {50256}

    # Text to insert at the beginning of each sequence
    BOS_TEXT = '\n '

    def __init__(self, vocab_path: str) -> None:
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=vocab_path)

        self.bos_token = self.tokenizer.bos_token_id
        self.filter_token_ids_cache = LRUCache(TOKENIZER_FILTER_TOKEN_IDS_CACHE_SIZE)
        self.TOKENIZER_WORD_TOKEN_SEPARATOR = set([idx for idx in range(len(self)) if self[idx][0] in TOKENIZER_WORD_TOKEN_SEPARATOR_SET])

    def __iter__(self) -> int:
        yield from self.tokenizer.vocab

    def __len__(self) -> int:
        return len(self.tokenizer.vocab)

    def __getitem__(self, idx: int) -> str:
        return self.tokenizer.decode(idx)

    @property
    def upper_tokens(self) -> Set:
        if not hasattr(self, '_upper_tokens'):
            self._upper_tokens = {idx for idx in range(len(self)) if any([c.isupper() and c not in TOKENIZER_WORD_TOKEN_SEPARATOR_SET for c in self[idx]])} # pylint: disable=attribute-defined-outside-init

        return self._upper_tokens

    def _filter_token_ids(self,
                          filter_prefix: str,
                          filter_token_ids_cache: LRUCache) -> Tuple[int, ...]:
        if len(filter_prefix) > 0 and filter_prefix[0] == ' ':
            filter_prefix = 'Ġ' + filter_prefix[1:]

        idx_token_result = None
        filter_prefix_len = len(filter_prefix)

        cached_filter_prefix = next((filter_prefix[:i] for i in reversed(range(1, len(filter_prefix) - 1)) \
            if filter_prefix[:i] in filter_token_ids_cache), None)

        if len(filter_prefix) > 0 and cached_filter_prefix is not None:
            prefilter_full_result = filter_token_ids_cache[cached_filter_prefix]
            idx_token_result = [(idx, token) for idx, token in prefilter_full_result \
                if token[:min(len(token), filter_prefix_len)] == filter_prefix[:min(len(token), filter_prefix_len)]]

        else:
            idx_token_result = [(idx, token) for token, idx in self.tokenizer.vocab.items() \
                if token[:min(len(token), filter_prefix_len)] == filter_prefix[:min(len(token), filter_prefix_len)]]

        filter_token_ids_cache[filter_prefix] = idx_token_result
        result = tuple(idx for idx, _ in idx_token_result)

        return result

    @functools.lru_cache(maxsize=128)
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, input_ids: List[int]) -> str:
        return self.tokenizer.decode(input_ids)

    @functools.lru_cache(maxsize=32768)
    def filter_token_mask_ids(self, filter_prefix: str) -> np.ndarray:
        if len(filter_prefix) == 0:
            return np.ones((len(self),), dtype=np.float32, order='C')

        filter_token_ids = list(self._filter_token_ids(filter_prefix, self.filter_token_ids_cache))

        mask = np.zeros((len(self),), dtype=np.float32, order='C')
        mask[filter_token_ids] = 1.0

        return mask

    @functools.lru_cache(maxsize=32768)
    def filter_token_ids(self, filter_prefix: str) -> Tuple[int, ...]:
        result = self._filter_token_ids(filter_prefix, self.filter_token_ids_cache)

        return result

    @functools.lru_cache(maxsize=32768)
    def filter_token_tuple_ids(self, filter_prefix: str) -> Tuple[int, ...]:
        result = self._filter_token_ids(filter_prefix, self.filter_token_ids_cache)
        result = tuple((idx,) for idx in result)

        return result

    def clean(self, text: str, add_bos_text: Optional[bool] = True) -> str:
        text = re.sub(r"[\u2010\u2011\u2012]", "-", text)
        text = re.sub(r"\s*[\u2013\u2014\u2015]\s*", " - ", text)
        text = re.sub(r"[\u2018\u2019\u201a\u201b\xb4]", "'", text)
        text = re.sub(r"[\u201c\u201d\u201e\u201f]", '"', text)
        text = REGEX_WHITESPACE.sub(" ", text)
        text = REGEX_NEW_LINE.sub("\n ", text)

        if add_bos_text:
            text = self.BOS_TEXT + text.lstrip()

        return text

    def find_context_prefix(self, text: str) -> Tuple[str, str]:
        m = REGEX_SPLIT.match(text)

        if m is None:
            context = ''
            prefix = text

        else:
            context = m.group(1)
            prefix = m.group(2)

        return (context, prefix)
