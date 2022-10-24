# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

from __future__ import annotations

from typing import Optional
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from archai.nlp.datasets.hf_datasets.tokenizer_utils.token_config import TokenConfig


class ArchaiPreTrainedTokenizerFast(PreTrainedTokenizerFast):
    """
    """

    def __init__(
        self,
        *args,
        tokenizer_file: Optional[str] = None,
        token_config_file: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        """

        if tokenizer_file is None:
            raise ValueError("`tokenizer_file` must be defined.")

        self.token_config = TokenConfig.from_file(token_config_file)
        kwargs["bos_token"] = self.token_config.bos_token
        kwargs["eos_token"] = self.token_config.eos_token
        kwargs["unk_token"] = self.token_config.unk_token
        kwargs["sep_token"] = self.token_config.sep_token
        kwargs["pad_token"] = self.token_config.pad_token
        kwargs["cls_token"] = self.token_config.cls_token
        kwargs["mask_token"] = self.token_config.mask_token

        super().__init__(*args, tokenizer_file=tokenizer_file, **kwargs)
