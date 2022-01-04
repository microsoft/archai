# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Corpus-related utilities that defines input data.
"""

from typing import List, Optional

from archai.common import utils
from archai.nlp.datasets.tokenizer_utils.special_token_enum import \
    SpecialTokenEnum


class TokenConfig:
    """Defines base configuration for a token.

    """

    def __init__(self,
                 bos_token: Optional[str] = '<|endoftext|>',
                 eos_token: Optional[str] = '<|endoftext|>',
                 unk_token: Optional[str] = '<|endoftext|>',
                 pad_token: Optional[str] = None,
                 add_prefix_space: Optional[bool] = False,
                 add_prefix_new_line: Optional[bool] = False,
                 lower_case: Optional[bool] = False) -> None:
        """Overrides initialization method.

        Args:
            bos_token: Begin-of-sentence token.
            eos_token: End-of-sentence token.
            unk_token: Unknown token.
            pad_token: Padding token.
            add_prefix_space: Whether to add a space prefix or not.
            add_prefix_new_line: Whether to add a new line prefix or not.
            lower_case: Whether to apply or not lower case to token.

        """

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.add_prefix_space = add_prefix_space
        self.add_prefix_new_line = add_prefix_new_line
        self.lower_case = lower_case

    def get_special_tokens(self) -> List[str]:
        """Gets the special tokens.

        Returns:
            (List[str]): Special tokens.

        """

        return utils.dedup_list([stok for stok in (self.unk_token, self.bos_token, self.eos_token, self.pad_token) if stok])

    def special_token_name(self, sp: SpecialTokenEnum) -> str:
        """Returns the token based on special token enumerator.

        Args:
            sp: Input special token.

        Returns:
            (str): Enumerated special token.
            
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
