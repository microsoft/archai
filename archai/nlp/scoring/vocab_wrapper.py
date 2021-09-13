import logging
import functools
from typing import Tuple

import numpy as np

from archai.common.lru_cache import LRUCache
from archai.nlp.scoring.scoring_utils import WORD_TOKEN_SEPARATOR_SET, RE_SPLIT
from archai.nlp.tokenizer_utils.vocab_base import VocabBase

TOKENIZER_FILTER_TOKEN_IDS_CACHE_SIZE = 65536

class VocabWrapper:
    # Set of tokens that should not be shown to the user
    INVALID_TOKENS = {50256}
    # Text to insert at the beginning of each sequence
    BOS_TEXT = ""

    def __init__(self, vocab:VocabBase):
        self.vocab = vocab
        self.idx2token = self.vocab.ids_to_tokens(list(range(len(self.vocab))))

        self.filter_token_ids_cache = LRUCache(TOKENIZER_FILTER_TOKEN_IDS_CACHE_SIZE)

        self.WORD_TOKEN_SEPARATOR_IDX = set([idx for idx in range(len(self)) if self[idx][0] in WORD_TOKEN_SEPARATOR_SET]) # pylint: disable=invalid-name
        logging.debug(f"len(WORD_TOKEN_SEPARATOR_IDX): {len(self.WORD_TOKEN_SEPARATOR_IDX)}")
        # self.warm_cache()

    def warm_cache(self):
        for prefix_len in range(1, 5):
            prefixes = set()
            for idx, token in enumerate(self):
                token_prefix = token[:prefix_len]
                prefixes.add(token_prefix)
                token_prefix_list = self.filter_token_ids_cache.get(token_prefix, [])
                token_prefix_list.append((idx, token))
                self.filter_token_ids_cache[token_prefix] = token_prefix_list

            for prefix in prefixes:
                self.filter_token_ids(prefix)
                self.filter_token_mask_ids(prefix)

    @functools.lru_cache(maxsize=32768)
    def filter_token_mask_ids(self, filter_prefix: str) -> np.ndarray:
        # start_time = time.time()
        if len(filter_prefix) == 0:
            return np.ones((len(self),), dtype=np.float32, order='C')

        filter_token_ids = list(self._filter_token_ids(filter_prefix, self.filter_token_ids_cache))
        mask = np.zeros((len(self),), dtype=np.float32, order='C')
        mask[filter_token_ids] = 1.0
        # print(f"filter_token_mask_ids: prefix: {filter_prefix} sum = {np.sum(mask)} time = {1000*(time.time() - start_time):.3f}")
        return mask

    @functools.lru_cache(maxsize=128)
    def encode(self, text: str) -> list:
        return self.vocab.encode_line(text)

    def decode(self, input_ids: list) -> str:
        return self.vocab.decode_line(input_ids)

    @functools.lru_cache(maxsize=32768)
    def filter_token_ids(self, filter_prefix: str) -> tuple:
        result = self._filter_token_ids(filter_prefix, self.filter_token_ids_cache)
        return result

    def find_context_prefix(self, text: str) -> Tuple[str, str]:
        """Split the text into prefix and context
        """
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

    def _filter_token_ids(self, filter_prefix: str, filter_token_ids_cache: LRUCache) -> tuple:
        """Return token ids that are consistent with the filtered prefix. Note that if prefix is:
            Micro
            it will return:
            [44, 13031, 15905, 25437, 41541]
            which corresponds to:
            ['M', 'Micro', 'Microsoft', 'Mic', 'Mi']
        """
        # TODO: , max_token_length_diff=100
        if len(filter_prefix) > 0 and filter_prefix[0] == ' ': # FIXME
            filter_prefix = 'Ġ' + filter_prefix[1:]


        idx_token_result = None
        filter_prefix_len = len(filter_prefix)

        # This part removed the cache
        # result = [idx for idx, token in enumerate(tokenizer) \
        #     if (filter_prefix_len - len(token)) < max_char_lookback \
        #         and token[:min(len(token), filter_prefix_len)] == filter_prefix[:min(len(token), filter_prefix_len)]]
        # return result

        cached_filter_prefix = next((filter_prefix[:i] for i in reversed(range(1, len(filter_prefix) - 1)) \
            if filter_prefix[:i] in filter_token_ids_cache), None)
        if len(filter_prefix) > 0 and cached_filter_prefix is not None: # filter_prefix[:-1] in self.filter_token_ids_cache:
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
    def UPPER_TOKENS(self) -> set: # pylint: disable=invalid-name
        """Return list of uppercase tokens"""
        if not hasattr(self, '_upper_tokens'):
            # c not in WORD_TOKEN_SEPARATOR_SET check is mostly to avoid putting words with space in the set
            self._upper_tokens = {idx for idx in range(len(self)) if any([c.isupper() and c not in WORD_TOKEN_SEPARATOR_SET for c in self[idx]])} # pylint: disable=attribute-defined-outside-init

        return self._upper_tokens

    @functools.lru_cache(maxsize=32768)
    def filter_token_tuple_ids(self, filter_prefix: str) -> tuple:
        """As above filter_token_ids, but returns a tuple of single element tuples for faster post-processing."""
        result = self._filter_token_ids(filter_prefix, self.filter_token_ids_cache)
        result = tuple((idx,) for idx in result)
        return result

    def clean(self, text: str, add_bos_text=True) -> str: # pylint: disable=unused-argument
        """Clean the text before tokenizing."""
        if add_bos_text:
            text = self.BOS_TEXT + text

        return text