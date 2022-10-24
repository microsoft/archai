# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

from __future__ import annotations

import json
from typing import List, Optional


SPECIAL_TOKENS = {
    "bos_token": "[BOS]",
    "eos_token": "[EOS]",
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]",
    "gpt2_eos_token": "<|endoftext|>",
    "transfo_xl_sep_token": "<formula>",
}


class TokenConfig:
    """Serves as the base foundation of a token's configuration."""

    def __init__(
        self,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        unk_token: Optional[str] = None,
        sep_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        cls_token: Optional[str] = None,
        mask_token: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initializes a token's configuration class by setting attributes.

        Args:
            bos_token: Begin-of-sentence token.
            eos_token: End-of-sentence token.
            unk_token: Unknown token.
            sep_token: Separator token (used for separating two sequences).
            pad_token: Padding token.
            cls_token: Input class token.
            mask_token: Masked token.

        """

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token

    @classmethod
    def from_file(cls: TokenConfig, token_config_path: str) -> TokenConfig:
        """Creates a class instance from an input file.

        Args:
            token_config_path: Path to the token's configuration file.

        Returns:
            (TokenConfig): Instance of the TokenConfig class.

        """

        try:
            with open(token_config_path, "r") as f:
                return cls(**json.load(f))
        except FileNotFoundError as error:
            raise error(f"{token_config_path} could not be found.")

    @property
    def special_tokens(self) -> List[str]:
        """Gathers the available special tokens.

        Returns:
            (List[str]): List of available special tokens.

        """

        return list(
            filter(
                None,
                [
                    self.bos_token,
                    self.eos_token,
                    self.unk_token,
                    self.sep_token,
                    self.pad_token,
                    self.cls_token,
                    self.mask_token,
                ],
            )
        )

    def save(self, token_config_path: str) -> None:
        """Saves the token's configuration to an output JSON file.

        Args:
            token_config_path: Path to where token's configuration should be saved.

        """

        with open(token_config_path, "w") as f:
            json.dump(self.__dict__, f)
