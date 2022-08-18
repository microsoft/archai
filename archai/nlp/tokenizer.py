# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Customizable tokenization utilities from huggingface/transformers.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from datasets.arrow_dataset import Dataset
from tokenizers import Tokenizer
from tokenizers.trainers import Trainer
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

# Available special tokens
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


class ArchaiTokenConfig:
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

        # Special tokens
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token

    @classmethod
    def from_file(cls: ArchaiTokenConfig, token_config_path: str) -> ArchaiTokenConfig:
        """Creates a class instance from an input file.

        Args:
            token_config_path: Path to the token's configuration file.

        Returns:
            (ArchaiTokenConfig): Instance of the ArchaiTokenConfig class.

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

    def save(self, output_token_config_path: str) -> None:
        """Saves the token's configuration to an output JSON file.

        Args:
            output_token_config_path: Path to where token's configuration should be saved.

        """

        with open(output_token_config_path, "w") as f:
            json.dump(self.__dict__, f)


class ArchaiTokenizer:
    """Serves as the base foundation of a tokenization pipeline."""

    def __init__(self, token_config: ArchaiTokenConfig, tokenizer: Tokenizer, trainer: Trainer) -> None:
        """Attaches required objects for the ArchaiTokenizer.

        Args:
            token_config: ArchaiTokenConfig class with token's configuration.
            tokenizer: Tokenizer class with model from huggingface/transformers.
            trainer: Trainer class from huggingface/transformers.

        """

        # Attaches input arguments as attributes
        self.token_config = token_config
        self.tokenizer = tokenizer
        self.trainer = trainer

    def get_vocab(self, with_added_tokens: Optional[bool] = True) -> Dict[str, int]:
        """Gets the tokenizer vocabulary.

        Args:
            with_added_tokens: Includes additional tokens that were added.

        Returns:
            (Dict[str, int]): Mapping between tokens' keys and values.

        """

        return self.tokenizer.get_vocab(with_added_tokens=with_added_tokens)

    def train_from_iterator(self, dataset: Dataset) -> None:
        """Trains from in-memory data.

        Args:
            dataset: Raw data to be tokenized.

        """

        def _batch_iterator(
            dataset: Dataset,
            batch_size: Optional[int] = 10000,
            column_name: Optional[str] = "text",
        ) -> Dataset:
            """Iterates over dataset to provide batches.

            Args:
                dataset: Dataset that should be iterated over.
                batch_size: Size of each batch.
                column_name: Name of column that should be retrieved.

            Yields:
                (Dataset): Batch of data based on size and `column_name`.

            """

            for i in range(0, len(dataset), batch_size):
                yield dataset[i : i + batch_size][column_name]

        return self.tokenizer.train_from_iterator(_batch_iterator(dataset), self.trainer, len(dataset))

    def save(self, output_tokenizer_path: str) -> None:
        """Saves the pre-trained tokenizer and token's configuration to disk.

        Args:
            output_tokenizer_path: Path to where tokenizer should be saved.

        """

        output_folder_path = os.path.dirname(output_tokenizer_path)
        output_token_config_path = os.path.join(output_folder_path, "token_config.json")

        if output_folder_path and not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        self.tokenizer.save(output_tokenizer_path)
        self.token_config.save(output_token_config_path)


class ArchaiPreTrainedTokenizer(AutoTokenizer):
    """Serves as an abstraction to load/use a pre-trained tokenizer."""

    def __init__(self, **kwargs) -> None:
        """Overrides with custom keyword arguments."""

        super().__init__(**kwargs)


class ArchaiPreTrainedTokenizerFast(PreTrainedTokenizerFast):
    """Serves as an abstraction to load/use a fast pre-trained tokenizer."""

    def __init__(self, *args, **kwargs) -> None:
        """Overrides with custom arguments and keyword arguments."""

        token_config_file = kwargs.pop("token_config_file", None)
        if token_config_file is None:
            self.token_config = ArchaiTokenConfig()
        else:
            self.token_config = ArchaiTokenConfig.from_file(token_config_file)

        # Fills up missing special tokens
        kwargs["bos_token"] = self.token_config.bos_token
        kwargs["eos_token"] = self.token_config.eos_token
        kwargs["unk_token"] = self.token_config.unk_token
        kwargs["sep_token"] = self.token_config.sep_token
        kwargs["pad_token"] = self.token_config.pad_token
        kwargs["cls_token"] = self.token_config.cls_token
        kwargs["mask_token"] = self.token_config.mask_token

        super().__init__(*args, **kwargs)
