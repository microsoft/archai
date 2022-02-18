# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Wraps Archai's vocabulary to work with Text Predictor.
"""

import functools
import logging
from typing import List, Optional, Set, Tuple

import numpy as np

from archai.common.lru_cache import LRUCache
from archai.nlp.datasets.tokenizer_utils.vocab_base import VocabBase
from archai.nlp.scoring_metrics.scoring_utils import RE_SPLIT, WORD_TOKEN_SEPARATOR_SET

# Token-related constants
TOKENIZER_FILTER_TOKEN_IDS_CACHE_SIZE = 65536

class VocabWrapper:
    """
    """

    # Set of tokens that should not be shown to the user
    INVALID_TOKENS = {50256}

    # Text to insert at the beginning of each sequence
    BOS_TEXT = ''

    def __init__(self, vocab: VocabBase) -> None:
        self.vocab = vocab

        self.idx2token = self.vocab.ids_to_tokens(list(range(len(self.vocab))))
        self.filter_token_ids_cache = LRUCache(TOKENIZER_FILTER_TOKEN_IDS_CACHE_SIZE)

        self.WORD_TOKEN_SEPARATOR_IDX = set([idx for idx in range(len(self)) if self[idx][0] in WORD_TOKEN_SEPARATOR_SET])

        logging.debug(f'WORD_TOKEN_SEPARATOR_IDX size: {len(self.WORD_TOKEN_SEPARATOR_IDX)}')

    @functools.lru_cache(maxsize=32768)
    def filter_token_mask_ids(self, filter_prefix: str) -> np.ndarray:
        if len(filter_prefix) == 0:
            return np.ones((len(self),), dtype=np.float32, order='C')

        filter_token_ids = list(self._filter_token_ids(filter_prefix, self.filter_token_ids_cache))

        mask = np.zeros((len(self),), dtype=np.float32, order='C')
        mask[filter_token_ids] = 1.0

        return mask

    @functools.lru_cache(maxsize=128)
    def encode(self, text: str) -> List[int]:
        return self.vocab.encode_text(text)

    def decode(self, input_ids: List[int]) -> str:
        return self.vocab.decode_text(input_ids)

    @functools.lru_cache(maxsize=32768)
    def filter_token_ids(self, filter_prefix: str) -> Tuple[int, ...]:
        result = self._filter_token_ids(filter_prefix, self.filter_token_ids_cache)

        return result

    def find_context_prefix(self, text: str) -> Tuple[str, str]:
        m = RE_SPLIT.match(text)

        if m is None:
            context = ''
            prefix = text

        else:
            context = m.group(1)
            prefix = m.group(2)

        return (context, prefix)

    def __iter__(self):
        yield from self.idx2token

    def __len__(self) -> int:
        return len(self.idx2token)

    def __getitem__(self, idx: int) -> str:
        return self.idx2token[idx]

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
            idx_token_result = [(idx, token) for idx, token in enumerate(self) \
                if token[:min(len(token), filter_prefix_len)] == filter_prefix[:min(len(token), filter_prefix_len)]]

        filter_token_ids_cache[filter_prefix] = idx_token_result
        result = tuple(idx for idx, _ in idx_token_result)

        return result

    @property
    def UPPER_TOKENS(self) -> Set:
        if not hasattr(self, '_upper_tokens'):
            self._upper_tokens = {idx for idx in range(len(self)) if any([c.isupper() and c not in WORD_TOKEN_SEPARATOR_SET for c in self[idx]])} # pylint: disable=attribute-defined-outside-init

        return self._upper_tokens

    @functools.lru_cache(maxsize=32768)
    def filter_token_tuple_ids(self, filter_prefix: str) -> Tuple[int, ...]:
        result = self._filter_token_ids(filter_prefix, self.filter_token_ids_cache)
        result = tuple((idx,) for idx in result)

        return result

    def clean(self, text: str, add_bos_text: Optional[bool] = True) -> str:
        if add_bos_text:
            text = self.BOS_TEXT + text

        return text
