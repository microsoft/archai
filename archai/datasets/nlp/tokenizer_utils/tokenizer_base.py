# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import List, Optional

import torch
from overrides import EnforceOverrides

from archai.common.logging_utils import get_logger
from archai.datasets.nlp.tokenizer_utils.token_config import SpecialTokenEnum

logger = get_logger(__name__)


class TokenizerBase(EnforceOverrides):
    """A customizable tokenization pipeline for encoding and decoding text.

    This is an abstract base class that defines the interface for implementing a
    vocabulary class that can be used for tokenization. Subclasses should override
    the abstract methods in this class to provide specific implementations.

    """

    @abstractmethod
    def __len__(self) -> int:
        """Get the length of the vocabulary.

        Returns:
            The length of the vocabulary.

        """

    @abstractmethod
    def train(self, filepaths: List[str]) -> None:
        """Train the tokenizer on a list of files.

        Args:
            filepaths: A list of paths to input files.

        """

    @abstractmethod
    def is_trained(self) -> bool:
        """Check if the vocabulary has been trained.

        Returns:
            `True` if the vocabulary has been trained, `False` otherwise.

        """

    @abstractmethod
    def load(self) -> None:
        """Load a pre-trained tokenizer."""

    @abstractmethod
    def encode_text(self, text: str) -> List[int]:
        """Encode text into tokens.

        Args:
            text: The input text to encode.

        Returns:
            The encoded text (tokens).

        """

    @abstractmethod
    def decode_text(self, ids: List[int]) -> str:
        """Decode tokens into text.

        Args:
            ids: The tokens to decode.

        Returns:
            The decoded tokens (text).

        """

    @abstractmethod
    def special_token_id(self, sp: SpecialTokenEnum) -> int:
        """Get the identifier of a special token.

        Args:
            sp: The special token's enumerator.

        Returns:
            The special token's identifier.

        """

    @abstractmethod
    def token_to_id(self, t: str) -> int:
        """Convert a string-based token to its identifier.

        Args:
            t: The string-based token.

        Returns:
            The token's identifier.

        """

    @abstractmethod
    def id_to_token(self, id: int) -> str:
        """Convert a token identifier to its string-based representation.

        Args:
            id: The token's identifier.

        Returns:
            The string-based token.

        """

    def tokens_to_ids(self, ts: List[str]) -> List[int]:
        """Convert a list of string-based tokens to their corresponding identifiers.

        Args:
            ts: A list of string-based tokens.

        Returns:
            The identifiers corresponding to the input tokens.

        """

        return [self.token_to_id(t) for t in ts]

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert a list of tokens' identifiers to their string-based representations.

        Args:
            ids: A list of tokens' identifiers.

        Returns:
            The string-based representations of the input tokens.

        """

        return [self.id_to_token(id) for id in ids]

    def encode_file(self, path: str, verbose: Optional[bool] = True) -> torch.Tensor:
        """Encode text from an input file.

        This method reads text from the specified file and encodes it using
        the `encode_text` method. It also includes options for verbosity and
        efficiently handling large datasets by converting the encoded tokens
        to a `torch.Tensor` every 500k lines.

        Args:
            path: The path to the input file.
            verbose: Whether to add verbosity to the logger.

        Returns:
            The encoded tokens.

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
