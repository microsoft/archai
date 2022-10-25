# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

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

        token_config = TokenConfig.from_file(token_config_file)
        token_config_dict = token_config.to_dict()

        super().__init__(*args, tokenizer_file=tokenizer_file, **token_config_dict, **kwargs)
