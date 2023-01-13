# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Corpus-related class for loading and encoding datasets."""

import glob
import os
from typing import Optional, Tuple, Union

import numpy as np
import torch

from archai.common import utils
from archai.nlp import logging_utils
from archai.nlp.datasets.nvidia import distributed_utils
from archai.nlp.datasets.nvidia.lm_iterators import (
    LMMultiFileIterator,
    LMOrderedIterator,
)
from archai.nlp.datasets.nvidia.tokenizer_utils.bbpe_vocab import BbpeVocab
from archai.nlp.datasets.nvidia.tokenizer_utils.gpt2_vocab import Gpt2Vocab
from archai.nlp.datasets.nvidia.tokenizer_utils.vocab_base import VocabBase
from archai.nlp.datasets.nvidia.tokenizer_utils.word_vocab import WordVocab

logger = logging_utils.get_logger(__name__)


class Corpus:
    """Creates and trains the vocabulary/tokenizer, loads the dataset and encodes the data."""

    def __init__(
        self,
        dataset: str,
        dataset_dir: str,
        cache_dir: str,
        vocab_type: str,
        vocab_size: Optional[int] = None,
        refresh_cache: Optional[bool] = False,
    ) -> None:
        """Initialize the `Corpus` class by defining attributes and creating
        cache-related paths.

        Args:
            dataset: Name of the dataset.
            dataset_dir: Path to the dataset folder.
            cache_dir: Path to the cache folder.
            vocab_type: Type of vocabulary/tokenizer.
                Valid options are `word`, `bbpe`, `gpt2`, or `bpe`.
            vocab_size: Vocabulary size.
            refresh_cache: Whether to refresh the cache.

        """

        self.dataset_dir = dataset_dir
        self.dataset = dataset
        self.vocab_type = vocab_type
        self.vocab_size = vocab_size

        # Corpus cache is created using dataset/vocab_type/vocab_size path
        self.corpus_cache_dir = utils.full_path(
            os.path.join(cache_dir, str(dataset), str(vocab_type), str(vocab_size)), create=True
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
        dataset: str, vocab_type: str, vocab_cache_dir: str, vocab_size: Optional[int] = None
    ) -> VocabBase:
        """Create the vocabulary.

        Args:
            dataset: Name of the dataset.
            vocab_type: Type of vocabulary.
            vocab_cache_dir: Path to the vocabulary cache folder.
            vocab_size: Vocabulary size.

        Returns:
            Vocabulary.

        """

        if vocab_type == "word":
            bos_token, eos_token, lower_case = None, "<eos>", False

            if dataset in ["wt103", "wt2"] or dataset.startswith("olx_"):
                pass
            elif dataset == "ptb":
                lower_case = True
            elif dataset == "lm1b":
                bos_token, eos_token = "<S>", "<S>"  # `<S>` is added for double EOS
            elif dataset in ["enwik8", "text8"]:
                eos_token, lower_case = None, True
            else:
                raise RuntimeError(f"Dataset: {dataset} is not supported yet.")

            vocab = WordVocab(
                save_path=vocab_cache_dir,
                vocab_size=vocab_size,
                bos_token=bos_token,
                eos_token=eos_token,
                lower_case=lower_case,
            )

        elif vocab_type == "bbpe":
            vocab = BbpeVocab(save_path=vocab_cache_dir, vocab_size=vocab_size or 50257)

        elif vocab_type == "gpt2":
            # Default vocab_size for GPT-2 is 50257
            vocab = Gpt2Vocab(save_path=vocab_cache_dir, vocab_size=vocab_size or 50257)

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
        if self.dataset in ["wt2", "wt103"]:
            train_file_name, valid_file_name, test_file_name = (
                "wiki.train.tokens",
                "wiki.valid.tokens",
                "wiki.test.tokens",
            )

        if self.dataset == "lm1b":
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

    def _create_train_vocab(self) -> VocabBase:
        """Create and trains the vocabulary.

        Returns:
            Pre-trained vocabulary.

        """

        self.vocab = Corpus._create_vocab(
            self.dataset, self.vocab_type, self.vocab_cache_dir, vocab_size=self.vocab_size
        )
        self._train_vocab()

        return self.vocab

    def _encode_files(self) -> None:
        """Encode dataset (training, validation and testing sets)."""

        train_filepath, valid_filepath, test_filepath = self._dataset_filepaths()

        if self.dataset == "lm1b":
            self.train = train_filepath
        else:
            self.train = self.vocab.encode_file(train_filepath)

        self.valid = self.vocab.encode_file(valid_filepath)
        self.test = self.vocab.encode_file(test_filepath)

    def train_and_encode(self) -> None:
        """Train the vocabulary/tokenizer and encodes the corpus."""

        logger.info(
            f"Training corpus: dataset = {self.dataset} | vocab_type = {self.vocab_type} | vocab_size = {self.vocab_size}"
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
            self.dataset, self.vocab_type, self.vocab_cache_dir, vocab_size=self.vocab_size
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

        utils.delete_file(self.train_cache_filepath)
        utils.delete_file(self.valid_cache_filepath)
        utils.delete_file(self.test_cache_filepath)

        return False

    def save_cache(self) -> None:
        """Save the cache."""

        assert self.vocab is not None and self.vocab.is_trained()

        np.save(self.train_cache_filepath, self.train.numpy())
        np.save(self.valid_cache_filepath, self.valid.numpy())
        np.save(self.test_cache_filepath, self.test.numpy())

    def get_iterator(
        self,
        split: str,
        batch_size: int,
        seq_len: int,
        device: str,
        mem_len: Optional[int] = 0,
        ext_len: Optional[int] = 0,
    ) -> Union[LMOrderedIterator, LMMultiFileIterator]:
        """Get an iterator based on current corpus.

        Args:
            split: Name of the split.
            batch_size: Batch size.
            seq_len: Sequence length.
            device: Device where iterator should be loaded on.
            mem_len: Length of memory (for Transformer-XL).
            ext_len: Length of extended context (for Transformer-XL).

        Returns:
            Iterator.

        """

        if split == "train":
            input_ids = self.train

            if self.dataset in ["wt2", "wt103"] or self.dataset.startswith("olx_"):
                iterator = LMOrderedIterator(
                    input_ids, batch_size, seq_len, device=device, mem_len=mem_len, ext_len=ext_len
                )
            elif self.dataset == "lm1b":
                iterator = LMMultiFileIterator(
                    input_ids,
                    self.vocab,
                    batch_size,
                    seq_len,
                    device=device,
                    mem_len=mem_len,
                    ext_len=ext_len,
                )
            else:
                raise RuntimeError(f"Dataset: {self.dataset} is not supported yet.")

        elif split in ["valid", "test"]:
            input_ids = self.valid if split == "valid" else self.test

            if self.dataset in ["wt2", "wt103", "lm1b"] or self.dataset.startswith("olx_"):
                iterator = LMOrderedIterator(
                    input_ids, batch_size, seq_len, device=device, ext_len=ext_len, mem_len=mem_len
                )
            else:
                raise RuntimeError(f"Dataset: {self.dataset} is not supported yet.")

        else:
            raise RuntimeError(f"Split: {split} is not supported yet.")

        return iterator


def load_corpus(
    dataset: str,
    dataset_dir: str,
    cache_dir: str,
    vocab_type: str,
    vocab_size: Optional[int] = None,
    refresh_cache=False,
) -> Corpus:
    """Load a pre-trained corpus if available, or pre-trains a new one.

    Args:
        dataset: Name of the dataset.
        dataset_dir: Dataset folder.
        cache_dir: Path to the cache folder.
        vocab_type: Type of vocabulary/tokenizer.
        vocab_size: Vocabulary size.
        refresh_cache: Whether cache should be refreshed.

    Returns:
        Corpus with pre-trained vocabulary and encoded data.

    """

    corpus = Corpus(dataset, dataset_dir, cache_dir, vocab_type, vocab_size=vocab_size, refresh_cache=refresh_cache)
    if not corpus.load():
        corpus.train_and_encode()

        with distributed_utils.sync_workers() as rank:
            if rank == 0 and dataset != "lm1b":
                corpus.save_cache()

    return corpus
