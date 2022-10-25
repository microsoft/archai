# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


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
    """
    """

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
        """
        """

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token

    @property
    def special_tokens(self) -> List[str]:
        """
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

    def to_dict(self) -> Dict[str, Any]:
        """
        """

        return self.__dict__

    def save(self, token_config_path: str) -> None:
        """
        """

        with open(token_config_path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_file(cls: TokenConfig, token_config_path: str) -> TokenConfig:
        """
        """

        if not token_config_path:
            return cls()

        try:
            with open(token_config_path, "r") as f:
                return cls(**json.load(f))
        except FileNotFoundError:
            return cls()
