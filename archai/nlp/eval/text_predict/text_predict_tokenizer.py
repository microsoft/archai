# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Text Predict-based tokenizer."""

import functools
import re
from typing import List, Optional, Set, Tuple, Union

from transformers.models.auto.tokenization_auto import AutoTokenizer

from archai.nlp.datasets.hf.tokenizer_utils.pre_trained_tokenizer import (
    ArchaiPreTrainedTokenizerFast,
)
from archai.nlp.eval.eval_utils import cached_property
from archai.nlp.eval.text_predict.text_predict_utils import LRUCache

SEPARATOR_TOKENS = "Ġ \nĊ\t\.;:,'\"`<>\(\)\{\}\[\]\|\!@\#\$\%\^\&\*=\+\?/\\_\-~"
SEPARATOR_TOKENS_SET = set(SEPARATOR_TOKENS)


class TextPredictTokenizer:
    """Wrapper for a tokenizer used in the Text Predict framework."""

    BOS_TEXT = "\n "
    FILTER_TOKENS_CACHE_SIZE = 65536
    INVALID_TOKENS = {50256}

    REGEX_SPLIT = re.compile("^(.*)([" + SEPARATOR_TOKENS + "].*)$", re.MULTILINE | re.DOTALL)
    REGEX_WHITESPACE = re.compile("[\xa0 \t\u2002-\u2006\u200B]+", re.MULTILINE | re.DOTALL)
    REGEX_NEW_LINE = re.compile("\s*[\r\n]+\s*", re.MULTILINE | re.DOTALL)

    def __init__(self, tokenizer: Union[AutoTokenizer, ArchaiPreTrainedTokenizerFast]) -> None:
        """Initialize a `TextPredictTokenizer`.

        Args:
            tokenizer: A pre-trained tokenizer.

        """

        self.tokenizer = tokenizer
        self.filter_tokens_cache = LRUCache(self.FILTER_TOKENS_CACHE_SIZE)

    def __iter__(self) -> int:
        """Provide an iterator over the tokenizer's vocabulary.

        Yields:
            Token identifier.

        """

        yield from self.tokenizer.vocab

    def __len__(self) -> int:
        """Return the length of vocabulary.

        Args:
            Length of vocabulary.

        """

        return len(self.tokenizer.vocab)

    def __getitem__(self, idx: int) -> str:
        """Return a string-based token based on identifier.

        Args:
            idx: Token identifier.

        Returns:
            Token.

        """

        return self.tokenizer.decode(idx)

    @property
    def bos_token_id(self) -> int:
        """Return the begin-of-sentence token identifier.

        Returns:
            Begin-of-sentence token identifier.

        """

        return self.tokenizer.bos_token_id

    @cached_property
    def separator_tokens(self) -> Set:
        """Compute the tokens separators.

        Returns:
            Token separators.

        """

        return set([i for i in range(len(self)) if self[i][0] in SEPARATOR_TOKENS_SET])

    @cached_property
    def upper_tokens(self) -> Set:
        """Compute the upper-cased tokens.

        Returns:
            Upper-cased tokens.

        """

        return set(
            [i for i in range(len(self)) if any([c.isupper() and c not in SEPARATOR_TOKENS_SET for c in self[i]])]
        )

    @functools.lru_cache(maxsize=128)
    def encode(self, text: str) -> List[int]:
        """Encode text with the tokenizer.

        Args:
            text: The text to be encoded.

        Returns:
            A list of encoded tokens.

        """

        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """Decode tokens with the tokenizer.

        Args:
            tokens: The tokens to be decoded.

        Returns:
            The decoded text.

        """

        return self.tokenizer.decode(tokens)

    def _filter_tokens(self, filter_prefix: str) -> Tuple[int, ...]:
        """Core computation to filter tokens according to the supplied prefix.

        Args:
            filter_prefix: The prefix to filter tokens.

        Returns:
            A list of filtered tokens.

        """

        if len(filter_prefix) > 0 and filter_prefix[0] == " ":
            filter_prefix = "Ġ" + filter_prefix[1:]

        filtered_tokens = None
        filter_prefix_len = len(filter_prefix)

        cached_filter_prefix = next(
            (
                filter_prefix[:i]
                for i in reversed(range(1, len(filter_prefix) - 1))
                if filter_prefix[:i] in self.filter_tokens_cache
            ),
            None,
        )

        if len(filter_prefix) > 0 and cached_filter_prefix is not None:
            pre_filtered_tokens = self.filter_tokens_cache[cached_filter_prefix]
            filtered_tokens = [
                (idx, token)
                for idx, token in pre_filtered_tokens
                if token[: min(len(token), filter_prefix_len)] == filter_prefix[: min(len(token), filter_prefix_len)]
            ]
        else:
            filtered_tokens = [
                (idx, token)
                for token, idx in self.tokenizer.vocab.items()
                if token[: min(len(token), filter_prefix_len)] == filter_prefix[: min(len(token), filter_prefix_len)]
            ]

        self.filter_tokens_cache[filter_prefix] = filtered_tokens
        filtered_tokens = tuple(idx for idx, _ in filtered_tokens)

        return filtered_tokens

    @functools.lru_cache(maxsize=32768)
    def filter_tokens(self, filter_prefix: str) -> Tuple[int, ...]:
        """Filter tokens according to the supplied prefix.

        Args:
            filter_prefix: The prefix to filter tokens.

        Returns:
            A list of filtered tokens.

        """

        filtered_tokens = self._filter_tokens(filter_prefix)

        return filtered_tokens

    def clean_text(self, text: str, add_bos_text: Optional[bool] = True) -> str:
        """Perform pre-processing to clean text.

        Args:
            text: The text to be cleaned.
            add_bos_text: Whether to add the begin-of-sentence text.

        Returns:
            The cleaned text.

        """

        text = re.sub(r"[\u2010\u2011\u2012]", "-", text)
        text = re.sub(r"\s*[\u2013\u2014\u2015]\s*", " - ", text)
        text = re.sub(r"[\u2018\u2019\u201a\u201b\xb4]", "'", text)
        text = re.sub(r"[\u201c\u201d\u201e\u201f]", '"', text)
        text = self.REGEX_WHITESPACE.sub(" ", text)
        text = self.REGEX_NEW_LINE.sub("\n ", text)

        if add_bos_text:
            text = self.BOS_TEXT + text.lstrip()

        return text

    def find_context_and_prefix(self, text: str) -> Tuple[str, str]:
        """Find context and prefix from input text.

        Args:
            text: The text to be processed.

        Returns:
            A tuple of context and prefix.

        """

        match_text = self.REGEX_SPLIT.match(text)

        if match_text is None:
            context = ""
            prefix = text
        else:
            context = match_text.group(1)
            prefix = match_text.group(2)

        return (context, prefix)
