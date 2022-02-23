# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Archai-based vocabularies that works with the Text Predictor.
"""

import functools
import logging
import json
import os
import re
from typing import List, Optional, Set, Tuple

from tokenizers import ByteLevelBPETokenizer
from tokenizers import Tokenizer

from transformers import AutoTokenizer
import numpy as np

from archai.common.lru_cache import LRUCache
from archai.nlp.datasets.tokenizer_utils.vocab_base import VocabBase
from archai.nlp.metrics.text_predict.text_predict_utils import RE_SPLIT, WORD_TOKEN_SEPARATOR_SET

# Token-related constants
TOKENIZER_FILTER_TOKEN_IDS_CACHE_SIZE = 65536
TOKENIZER_BOS = '_BOS_'
TOKENIZER_MAPPING = {'<|unk|>': '_OOV_', '<|endoftext|>': TOKENIZER_BOS}
RE_WHITESPACE = re.compile("[\xa0 \t\u2002-\u2006\u200B]+", re.MULTILINE | re.DOTALL)
# This RE_NEW_LINE should (hopefully) make new line marker platform-independent
RE_NEW_LINE = re.compile("\s*[\r\n]+\s*", re.MULTILINE | re.DOTALL) # pylint: disable=anomalous-backslash-in-string



class VocabWrapper:
    """Wraps a VocabBase to comply with Text Preditor.

    """

    # Set of tokens that should not be shown to the user
    INVALID_TOKENS = {50256}

    # Text to insert at the beginning of each sequence
    BOS_TEXT = ''

    def __init__(self, vocab_path: str) -> None:
        # merges_file, vocab_json_file, _ = self._convert_to_separate_files(vocab_path)

        self.tokenizer = AutoTokenizer.from_pretrained(vocab_path)


        print(dir(self.tokenizer))
        # print(self.tokenizer.vocab)

        self.bos_token = self.tokenizer.bos_token_id
        self.filter_token_ids_cache = LRUCache(TOKENIZER_FILTER_TOKEN_IDS_CACHE_SIZE)
        
        self.WORD_TOKEN_SEPARATOR_IDX = set([idx for idx in range(len(self)) if self[idx] in WORD_TOKEN_SEPARATOR_SET])

        # logging.debug(f'WORD_TOKEN_SEPARATOR_IDX size: {len(self.WORD_TOKEN_SEPARATOR_IDX)}')

    def _convert_to_separate_files(self, vocab_path: str) -> Tuple[str, str]:
        vocab_dir = os.path.dirname(vocab_path)
        merges_path = f'{vocab_dir}/merges.txt'
        vocab_json_path = f'{vocab_dir}/vocab.json'
        vocab_txt_path = f'{vocab_dir}/vocab.txt'

        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        # Saves the vocab.json
        with open(vocab_json_path, 'w', encoding='utf-8') as f:
	        json.dump(vocab['model']['vocab'], f)

        # Saves the merges.json
        with open(merges_path, 'w', encoding='utf-8') as f:
            for i, merge in enumerate(vocab['model']['merges']):
                f.write(merge)
                if i != len(vocab['model']['merges']) - 1:
                    f.write('\n')

        # Saves the vocab.txt
        with open(vocab_txt_path, 'w', encoding='utf-8') as f:
            vocab_list = [0] * len(vocab['model']['vocab'])

            for token, idx in vocab['model']['vocab'].items():
                if token in TOKENIZER_MAPPING:
                    token = TOKENIZER_MAPPING[token]
                vocab_list[idx] = token

            for i, token in enumerate(vocab_list):
                f.write('%s' % (token))
                
                if i != len(vocab_list) - 1:
                    f.write('\n')

        return merges_path, vocab_json_path, vocab_txt_path

    def clean(self, text: str, add_bos_text=True) -> str:
        """Clean the text before tokenizing."""
        # One could code it up similar to this in C++:
        # https://stackoverflow.com/questions/40690460/python-removing-extra-special-unicode-characters
        # logging.debug(f"tokenizer.clean.before: '{text}'")
        # UnicodeHyphen,"[\u2014\u2015]",local.replace,,"-"
        text = re.sub(r"[\u2010\u2011\u2012]", "-", text)
        text = re.sub(r"\s*[\u2013\u2014\u2015]\s*", " - ", text)
        # UnicodeQuote,"[\u2018\u2019\xb4]",local.replace,,"'"
        text = re.sub(r"[\u2018\u2019\u201a\u201b\xb4]", "'", text)
        # UnicodeDoubleQuote,"[\u201c\u201d]",local.replace,,""""
        text = re.sub(r"[\u201c\u201d\u201e\u201f]", '"', text)

        text = RE_WHITESPACE.sub(" ", text)
        text = RE_NEW_LINE.sub("\n ", text)
        if add_bos_text:
            text = self.BOS_TEXT + text.lstrip()
        logging.debug(f"tokenizer.clean.after {add_bos_text}: '{text}'")
        return text

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
        print(f'encode: {text} -> {self.tokenizer.encode(text)}')
        return self.tokenizer.encode(text)

    def decode(self, input_ids: List[int]) -> str:
        return self.tokenizer.decode(input_ids)

    @functools.lru_cache(maxsize=32768)
    def filter_token_ids(self, filter_prefix: str) -> Tuple[int, ...]:
        result = self._filter_token_ids(filter_prefix, self.filter_token_ids_cache)

        return result

    def find_context_prefix(self, text: str) -> Tuple[str, str]:
        m = RE_SPLIT.match(text)

        # print(text, m)

        if m is None:
            context = ''
            prefix = text

        else:
            context = m.group(1)
            prefix = m.group(2)

        return (context, prefix)

    def __iter__(self) -> int:
        yield from self.tokenizer.vocab

    def __len__(self) -> int:
        return len(self.tokenizer.vocab)

    def __getitem__(self, idx: int) -> str:
        return self.tokenizer.decode(idx)

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
    def upper_tokens(self) -> Set:
        if not hasattr(self, '_upper_tokens'):
            self._upper_tokens = {idx for idx in range(len(self)) if any([c.isupper() and c not in WORD_TOKEN_SEPARATOR_SET for c in self[idx]])} # pylint: disable=attribute-defined-outside-init

        return self._upper_tokens

    @functools.lru_cache(maxsize=32768)
    def filter_token_tuple_ids(self, filter_prefix: str) -> Tuple[int, ...]:
        result = self._filter_token_ids(filter_prefix, self.filter_token_ids_cache)
        result = tuple((idx,) for idx in result)

        return result
