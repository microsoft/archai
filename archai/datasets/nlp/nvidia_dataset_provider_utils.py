# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import glob
import os
from typing import Optional, Tuple

import numpy as np
import torch

from archai.common.file_utils import get_full_path
from archai.common.logger import Logger
from archai.datasets.nlp.tokenizer_utils.bbpe_tokenizer import BbpeTokenizer
from archai.datasets.nlp.tokenizer_utils.gpt2_tokenizer import Gpt2Tokenizer
from archai.datasets.nlp.tokenizer_utils.tokenizer_base import TokenizerBase
from archai.datasets.nlp.tokenizer_utils.word_tokenizer import WordTokenizer

logger = Logger(source=__name__)


def delete_file(file_path: str) -> bool:
    """Delete a file.

    Args:
        file_path: Path to the file.

    Returns:
        `True` if the file was deleted, `False` otherwise.

    """

    if os.path.isfile(file_path):
        os.remove(file_path)
        return True

    return False


class Corpus:
    """Creates and trains the vocabulary/tokenizer, loads the dataset and encodes the data."""

    def __init__(
        self,
        dataset_name: str,
        dataset_dir: str,
        cache_dir: str,
        vocab_type: str,
        vocab_size: Optional[int] = None,
        refresh_cache: Optional[bool] = False,
    ) -> None:
        """Initialize the `Corpus` class by defining attributes and creating
        cache-related paths.

        Args:
            dataset_name: Name of the dataset.
            dataset_dir: Path to the dataset folder.
            cache_dir: Path to the cache folder.
            vocab_type: Type of vocabulary/tokenizer.
                Valid options are `word`, `bbpe`, `gpt2`, or `bpe`.
            vocab_size: Vocabulary size.
            refresh_cache: Whether to refresh the cache.

        """

        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.vocab_type = vocab_type
        self.vocab_size = vocab_size

        # Corpus cache is created using dataset/vocab_type/vocab_size path
        self.corpus_cache_dir = get_full_path(
            os.path.join(cache_dir, str(dataset_name), str(vocab_type), str(vocab_size)), create=True
        )

        # Encoded dataset (.npy files) cache paths
        self.train_cache_filepath = os.path.join(self.corpus_cache_dir, "train.npy")
        self.valid_cache_filepath = os.path.join(self.corpus_cache_dir, "valid.npy")
        self.test_cache_filepath = os.path.join(self.corpus_cache_dir, "test.npy")

        # Tokenizer-related files cache paths
        self.vocab_cache_dir = os.path.join(self.corpus_cache_dir, "vocab")
        self.refresh_cache = refresh_cache

        if refresh_cache:
            logger.info("Refreshing cache ...")

        self._clear_cache()

    @staticmethod
    def _create_vocab(
        dataset_name: str, vocab_type: str, vocab_cache_dir: str, vocab_size: Optional[int] = None
    ) -> TokenizerBase:
        """Create the vocabulary.

        Args:
            dataset_name: Name of the dataset.
            vocab_type: Type of vocabulary.
            vocab_cache_dir: Path to the vocabulary cache folder.
            vocab_size: Vocabulary size.

        Returns:
            Vocabulary.

        """

        if vocab_type == "word":
            bos_token, eos_token, lower_case = None, "<eos>", False

            if dataset_name in ["wt103", "wt2"] or dataset_name.startswith("olx_"):
                pass
            elif dataset_name == "ptb":
                lower_case = True
            elif dataset_name == "lm1b":
                bos_token, eos_token = "<S>", "<S>"  # `<S>` is added for double EOS
            elif dataset_name in ["enwik8", "text8"]:
                eos_token, lower_case = None, True
            else:
                raise RuntimeError(f"Dataset: {dataset_name} is not supported yet.")

            vocab = WordTokenizer(
                save_path=vocab_cache_dir,
                vocab_size=vocab_size,
                bos_token=bos_token,
                eos_token=eos_token,
                lower_case=lower_case,
            )

        elif vocab_type == "bbpe":
            vocab = BbpeTokenizer(save_path=vocab_cache_dir, vocab_size=vocab_size or 50257)

        elif vocab_type == "gpt2":
            # Default vocab_size for GPT-2 is 50257
            vocab = Gpt2Tokenizer(save_path=vocab_cache_dir, vocab_size=vocab_size or 50257)

        else:
            raise RuntimeError(f"Vocabulary: {vocab_type} is not supported yet.")

        return vocab

    def _clear_cache(self) -> None:
        """Clear the cache."""

        self.train = self.valid = self.test = self.vocab = None

    def _dataset_filepaths(self) -> Tuple[str, str, str]:
        """Get the dataset's file paths.

        Returns:
            Training, validation and testing file paths.

        """

        train_file_name, valid_file_name, test_file_name = "train.txt", "valid.txt", "test.txt"
        if self.dataset_name in ["wt2", "wt103"]:
            train_file_name, valid_file_name, test_file_name = (
                "wiki.train.tokens",
                "wiki.valid.tokens",
                "wiki.test.tokens",
            )

        if self.dataset_name == "lm1b":
            train_path_pattern = os.path.join(
                self.dataset_dir,
                "1-billion-word-language-modeling-benchmark-r13output",
                "training-monolingual.tokenized.shuffled",
                "news.en-*",
            )
            train_file_name_path = glob.glob(train_path_pattern)
        else:
            train_file_name_path = os.path.join(self.dataset_dir, train_file_name)
        valid_file_name_path = os.path.join(self.dataset_dir, valid_file_name)
        test_file_name_path = os.path.join(self.dataset_dir, test_file_name)

        return (
            train_file_name_path,
            valid_file_name_path,
            test_file_name_path,
        )

    def _train_vocab(self) -> None:
        """Train the vocabulary."""

        # If vocabulary cache does not exist
        if self.refresh_cache or not self.vocab.is_trained():
            logger.info("Training vocabulary ...")

            train_filepath, _, _ = self._dataset_filepaths()
            if not isinstance(train_filepath, list):
                train_filepath = [train_filepath]

            self.vocab.train(train_filepath)
            logger.info("Vocabulary trained.")

        else:
            self.vocab.load()
            logger.debug(f"Loading vocabulary ({self.vocab_type}, {self.vocab_size}) from: {self.vocab_cache_dir}")

    def _create_train_vocab(self) -> TokenizerBase:
        """Create and trains the vocabulary.

        Returns:
            Pre-trained vocabulary.

        """

        self.vocab = Corpus._create_vocab(
            self.dataset_name, self.vocab_type, self.vocab_cache_dir, vocab_size=self.vocab_size
        )
        self._train_vocab()

        return self.vocab

    def _encode_files(self) -> None:
        """Encode dataset (training, validation and testing sets)."""

        train_filepath, valid_filepath, test_filepath = self._dataset_filepaths()

        if self.dataset_name == "lm1b":
            self.train = train_filepath
        else:
            self.train = self.vocab.encode_file(train_filepath)

        self.valid = self.vocab.encode_file(valid_filepath)
        self.test = self.vocab.encode_file(test_filepath)

    def train_and_encode(self) -> None:
        """Train the vocabulary/tokenizer and encodes the corpus."""

        logger.info(
            f"Training corpus: dataset = {self.dataset_name} | vocab_type = {self.vocab_type} | vocab_size = {self.vocab_size}"
        )

        self._create_train_vocab()
        self._encode_files()

        train_size = f"{len(self.train)} files" if isinstance(self.train, list) else self.train.size(0)
        logger.debug(f"Size: train = {train_size} | valid = {self.valid.size(0)} | test = {self.test.size(0)}")

    def load(self) -> bool:
        """Load a pre-trained corpus.

        Returns:
            Whether pre-trained corpus has been successfully loaded.

        """

        # Ensures tokenizer cache is loaded as well
        self.vocab = Corpus._create_vocab(
            self.dataset_name, self.vocab_type, self.vocab_cache_dir, vocab_size=self.vocab_size
        )

        cache_exists = (
            os.path.exists(self.train_cache_filepath)
            and os.path.exists(self.valid_cache_filepath)
            and os.path.exists(self.test_cache_filepath)
        )

        # If .npy files exists, corpus cache is available
        if not self.refresh_cache and cache_exists and self.vocab is not None and self.vocab.is_trained():
            logger.info(f"Loading cache from: {self.train_cache_filepath} ...")

            self.vocab.load()

            self.train = torch.from_numpy(np.load(self.train_cache_filepath))
            self.valid = torch.from_numpy(np.load(self.valid_cache_filepath))
            self.test = torch.from_numpy(np.load(self.test_cache_filepath))

            logger.debug(
                f"Size: train = {self.train.size(0)} | valid = {self.valid.size(0)} | test = {self.test.size(0)}"
            )

            return True

        logger.info("Clearing and rebuilding cache ...")
        self._clear_cache()

        delete_file(self.train_cache_filepath)
        delete_file(self.valid_cache_filepath)
        delete_file(self.test_cache_filepath)

        return False

    def save_cache(self) -> None:
        """Save the cache."""

        assert self.vocab is not None and self.vocab.is_trained()

        np.save(self.train_cache_filepath, self.train.numpy())
        np.save(self.valid_cache_filepath, self.valid.numpy())
        np.save(self.test_cache_filepath, self.test.numpy())
