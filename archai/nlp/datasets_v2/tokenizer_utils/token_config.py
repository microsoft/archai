# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

from typing import Optional


class TokenConfig:
    """
    """

    def __init__(self,
                 bos_token: Optional[str] = None,
                 eos_token: Optional[str] = None,
                 unk_token: Optional[str] = None,
                 pad_token: Optional[str] = None,
                 mask_token: Optional[str] = None,
                 add_prefix_space: Optional[bool] = False,
                 add_prefix_new_line: Optional[bool] = False,
                 lower_case: Optional[bool] = False) -> None:
        # Special tokens
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.mask_token = mask_token

        # Additional configuration
        self.add_prefix_space = add_prefix_space
        self.add_prefix_new_line = add_prefix_new_line
        self.lower_case = lower_case

    @property
    def special_tokens(self):
        return list(filter(None, [self.bos_token,
                                  self.eos_token,
                                  self.unk_token,
                                  self.pad_token,
                                  self.mask_token]))
