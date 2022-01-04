# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Base vocabulary.
"""

import logging
from abc import abstractmethod
from collections import abc
from typing import List, Optional

import torch
from archai.nlp.datasets.tokenizer_utils.special_token_enum import \
    SpecialTokenEnum
from overrides import EnforceOverrides, overrides


class VocabBase(EnforceOverrides, abc.Sized):
    """Base definitions of a vocabulary.

    """

    @abstractmethod
    def train(self, filepaths: List[str]) -> None:
        """Trains a new vocabulary.

        Args:
            filepaths: Path to the input files.

        """

        pass

    @abstractmethod
    def load(self) -> None:
        """Loads a pre-trained vocabulary.
        """

        pass

    @abstractmethod
    def encode_text(self, text: str) -> List[int]:
        """Encodes input text into a list of tokens.

        Args:
            text: Input text.

        Returns:
            List[int]: Encoded tokens.

        """

        pass

    @abstractmethod
    def decode_text(self, ids: List[int]) -> str:
        """Decodes tokens into a string-based text.

        Args:
            ids: Tokens identifiers.

        Returns:
            (str): Decoded tokens.

        """

        pass

    @abstractmethod
    def is_trained(self) -> bool:
        """Checks whether vocabulary has already been trained.

        Returns:
            (bool): Whether vocabulary has already been trained.
            
        """

        pass

    @abstractmethod
    @overrides
    def __len__(self) -> int:
        """Calculates the size of the vocabulary.

        Args:
            (int): Size of vocabulary.

        """

        pass

    @abstractmethod
    def token_to_id(self, t: str) -> int:
        """Converts string-based token to identifier.

        Args:
            t: String-based token.

        Returns:
            (int): Token identifier.
            
        """

        pass

    @abstractmethod
    def id_to_token(self, id: int) -> str:
        """Converts token identifier to string-based token.

        Args:
            id: Token identifier.

        Returns:
            (str): String-based token.

        """

        pass

    @abstractmethod
    def special_token_id(self, sp: SpecialTokenEnum) -> int:
        """Gets the special token identifier.

        Args:
            sp: Special token enumerator.

        Returns:
            (int): Special token identifier.

        """

        pass

    def tokens_to_ids(self, ts: List[str]) -> List[int]:
        """Converts a list of string-based tokens to their identifiers.

        Args:
            ts: List of string-based tokens.

        Returns:
            (List[int]): List of tokens identifiers.

        """

        return [self.token_to_id(t) for t in ts]

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Converts a list of tokens identifiers to their string-based versions.

        Args:
            ids: List of tokens identifiers.

        Returns:
            (List[int]): List of string-based tokens.

        """

        return [self.id_to_token(id) for id in ids]

    def encode_file(self, path: str, verbose: Optional[bool] = True) -> List[int]:
        """Encodes a file.

        Args:
            path: File path.
            verbose: Whether to display additional logging or not.
        
        Returns:
            (List[int]): Encoded file.

        """

        logging.info(f'Encoding file: {path}')
        encoded = []
        tensor_encoded = torch.LongTensor()

        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                # Converts to tensor.Tensor every 500k lines otherwise Python list uses a lot of RAM
                if idx > 0 and idx % 500000 == 0:
                    tensor_encoded = torch.cat((tensor_encoded, torch.LongTensor(encoded)))
                    encoded = []

                if verbose and idx > 0 and idx % 500000 == 0:
                    logging.info(f'    completed file line {format(idx)}')

                tokens = self.encode_text(line)
                encoded.extend(tokens)

        if len(encoded) > 0:
            tensor_encoded = torch.cat((tensor_encoded, torch.LongTensor(encoded)))

        return tensor_encoded
