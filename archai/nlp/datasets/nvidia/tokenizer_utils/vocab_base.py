# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Utilities for tokenization pipelines with huggingface/tokenizers.
"""

from abc import abstractmethod
from collections import abc
from typing import List, Optional

import torch
from overrides import EnforceOverrides, overrides

from archai.nlp import logging_utils
from archai.nlp.datasets.nvidia.tokenizer_utils.special_token_enum import (
    SpecialTokenEnum,
)

logger = logging_utils.get_logger(__name__)


class VocabBase(EnforceOverrides, abc.Sized):
    """Implements a base class for a customizable tokenization pipeline."""

    @abstractmethod
    @overrides
    def __len__(self) -> int:
        """Length of the vocabulary.

        Returns:
            (int): Length of the vocabulary.

        """

        pass

    @abstractmethod
    def train(self, filepaths: List[str]) -> None:
        """Trains tokenizer from a list of files.

        Args:
            filepaths: List of paths to input files.

        """

        pass

    @abstractmethod
    def is_trained(self) -> bool:
        """Checks whether vocabulary has been trained.

        Returns:
            (bool): Whether vocabulary has been trained.

        """

        pass

    @abstractmethod
    def load(self) -> None:
        """Loads pre-trained tokenizer."""

        pass

    @abstractmethod
    def encode_text(self, text: str) -> List[int]:
        """Encodes text into tokens.

        Args:
            text: Input text.

        Returns:
            (List[int]): Encoded text (tokens).

        """

        pass

    @abstractmethod
    def decode_text(self, ids: List[int]) -> str:
        """Decodes tokens into text.

        Args:
            ids: Tokens.

        Returns:
            (str): Decoded tokens (text).

        """

        pass

    @abstractmethod
    def special_token_id(self, sp: SpecialTokenEnum) -> int:
        """Gets the identifier of special token.

        Args:
            sp: Special token's enumerator.

        Returns:
            (int): Special token's identifier.

        """

        pass

    @abstractmethod
    def token_to_id(self, t: str) -> int:
        """Converts a string-based token to its identifier.

        Args:
            t: String-based token.

        Returns:
            (int): Token's identifier.

        """

        pass

    @abstractmethod
    def id_to_token(self, id: int) -> str:
        """Converts a token identifier to its string-based representation.

        Args:
            id: Token's identifier.

        Returns:
            (str): String-based token.

        """

        pass

    def tokens_to_ids(self, ts: List[str]) -> List[int]:
        """Converts a set of string-based tokens to their identifiers.

        Args:
            ts: String-based tokens.

        Returns:
            (List[int]): Tokens' identifiers.

        """

        return [self.token_to_id(t) for t in ts]

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Converts a set of tokens' identifiers to their string-based representations.

        Args:
            ids: Tokens' identifiers.

        Returns:
            (List[str]): String-based tokens.

        """

        return [self.id_to_token(id) for id in ids]

    def encode_file(self, path: str, verbose: Optional[bool] = True) -> torch.Tensor:
        """Encodes text from an input file.

        Args:
            path: Input file.
            verbose: Whether should add verbosity to logger.

        Returns:
            (torch.Tensor): Encoded tokens.

        """

        logger.info(f"Encoding file: {path}")

        encoded = []
        tensor_encoded = torch.LongTensor()

        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                # Converts to tensor.Tensor every 500k lines,
                # otherwise Python list uses a lot of RAM
                if idx > 0 and idx % 500000 == 0:
                    tensor_encoded = torch.cat((tensor_encoded, torch.LongTensor(encoded)))
                    encoded = []

                if verbose and idx > 0 and idx % 500000 == 0:
                    logger.debug(f"Completed line: {format(idx)}")

                tokens = self.encode_text(line)
                encoded.extend(tokens)

        if len(encoded) > 0:
            tensor_encoded = torch.cat((tensor_encoded, torch.LongTensor(encoded)))

        return tensor_encoded
