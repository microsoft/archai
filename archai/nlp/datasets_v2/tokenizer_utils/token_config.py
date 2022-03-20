# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Token-related configuration, such as special tokens, pre-processing variables
    and additional information that must be included in the tokenization pipeline.
"""

from typing import List, Optional


class TokenConfig:
    """Token-related configuration, mostly for special tokens and
        additional tokenization-related information.

    """

    def __init__(self,
                 bos_token: Optional[str] = None,
                 eos_token: Optional[str] = None,
                 unk_token: Optional[str] = None,
                 sep_token: Optional[str] = None,
                 pad_token: Optional[str] = None,
                 cls_token: Optional[str] = None,
                 mask_token: Optional[str] = None,
                 model_max_length: Optional[int] = None,
                 add_prefix_space: Optional[bool] = False,
                 add_prefix_new_line: Optional[bool] = False,
                 lower_case: Optional[bool] = False) -> None:
        # Special tokens
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token

        # Additional configuration
        self.model_max_length = model_max_length
        self.add_prefix_space = add_prefix_space
        self.add_prefix_new_line = add_prefix_new_line
        self.lower_case = lower_case

    @property
    def special_tokens(self) -> List[str]:
        return list(filter(None, [self.bos_token,
                                  self.eos_token,
                                  self.unk_token,
                                  self.sep_token,
                                  self.pad_token,
                                  self.cls_token,
                                  self.mask_token]))

    def pre_process(self, token: str) -> str:
        if self.add_prefix_space:
            token = ' ' + token
        if self.add_prefix_new_line:
            token = '\n' + token
        if self.lower_case:
            token = token.lower()

        return token
