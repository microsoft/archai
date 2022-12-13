# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Utilities for defining tokens' configuration.
"""

from typing import List, Optional

from archai.common import utils
from archai.nlp.datasets.nvidia.tokenizer_utils.special_token_enum import (
    SpecialTokenEnum,
)


class TokenConfig:
    """Implements a base class for tokens' configuration."""

    def __init__(
        self,
        bos_token: Optional[str] = "<|endoftext|>",
        eos_token: Optional[str] = "<|endoftext|>",
        unk_token: Optional[str] = "<|endoftext|>",
        pad_token: Optional[str] = None,
        add_prefix_space: Optional[bool] = False,
        add_prefix_new_line: Optional[bool] = False,
        lower_case: Optional[bool] = False,
    ) -> None:
        """Initializes class by setting attributes.

        Args:
            bos_token: Begin-of-sentence token.
            eos_token: End-of-sentence token.
            unk_token: Unknown token.
            pad_token: Padding token.
            add_prefix_space: Whether a prefix space token should be added.
            add_prefix_new_line: Whether a prefix new line token should be added.
            lower_case: Whether lower case should be applied.

        """

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.add_prefix_space = add_prefix_space
        self.add_prefix_new_line = add_prefix_new_line
        self.lower_case = lower_case

    def get_special_tokens(self) -> List[str]:
        """Gets available special tokens.

        Returns:
            (List[str]): Special tokens.

        """

        return utils.dedup_list(
            [stok for stok in (self.unk_token, self.bos_token, self.eos_token, self.pad_token) if stok]
        )

    def special_token_name(self, sp: SpecialTokenEnum) -> str:
        """Gets the name of the special token.

        Args:
            sp: Special token enumerator.

        Returns:
            (str): Special token name.

        """

        if sp == SpecialTokenEnum.BOS:
            return self.bos_token
        if sp == SpecialTokenEnum.EOS:
            return self.eos_token
        if sp == SpecialTokenEnum.UNK:
            return self.unk_token
        if sp == SpecialTokenEnum.PAD:
            return self.pad_token

        return None
