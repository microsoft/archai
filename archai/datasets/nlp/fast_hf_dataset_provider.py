# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from hashlib import sha1
from pathlib import Path
from typing import List, Optional

import numpy as np
from datasets import load_dataset as hf_load_dataset
from datasets.dataset_dict import DatasetDict
from overrides import overrides
from transformers import AutoTokenizer

from archai.api.dataset_provider import DatasetProvider
from archai.common.ordered_dict_logger import OrderedDictLogger
from archai.datasets.nlp.fast_hf_dataset_provider_utils import (
    FastHfDataset,
    process_with_memory_map_files,
    process_with_shared_memory,
)
from archai.datasets.nlp.hf_dataset_provider_utils import tokenize_concatenated_dataset

logger = OrderedDictLogger(source=__name__)

if sys.version_info.major == 3 and sys.version_info.minor >= 8:
    ALLOW_SHARED_MEMORY = True
else:
    logger.warn("Shared memory is not available in Python < 3.8.")
    ALLOW_SHARED_MEMORY = False


class FastHfDatasetProvider(DatasetProvider):
    """Fast Hugging Face-based dataset provider."""

    def __init__(
        self,
        dataset: Optional[str] = "wikitext",
        subset: Optional[str] = "wikitext-2-raw-v1",
        tokenizer: Optional[str] = "gpt2",
        mapping_column_name: Optional[List[str]] = None,
        cache_dir: Optional[str] = "cache",
        validation_split: Optional[float] = 0.1,
        seed: Optional[int] = 42,
        num_workers: Optional[int] = 1,
        use_shared_memory: Optional[bool] = True,
    ) -> None:
        """Initialize Fast Hugging Face-based dataset provider.

        The initialization consists in pre-loading the dataset, encoding it
        using the specified tokenizer, and saving it to the cache directory.

        Args:
            dataset: Name of the dataset.
            subset: Name of the dataset configuration.
            tokenizer: Name of the tokenizer.
            mapping_column_name: The columns in `dataset` that should be tokenized.
            cache_dir: Path to the read/write cache directory.
            validation_split: Fraction of the dataset to use for validation.
            seed: Random seed.
            num_workers: Number of workers to use for encoding.
            use_shared_memory: Whether to use shared memory for caching.

        """

        super().__init__()

        self.dataset = dataset
        self.subset = subset
        self.tokenizer = tokenizer
        self.mapping_column_name = mapping_column_name
        self.validation_split = validation_split
        self.seed = seed
        self.num_workers = num_workers
        self.use_shared_memory = use_shared_memory and ALLOW_SHARED_MEMORY
        self.cache_dir = Path(cache_dir) / self.fingerprint

        # If cache is not available, encode the dataset and save it to cache
        if not self.cache_dir.is_dir():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._encode_dataset()

    @property
    def fingerprint(self) -> str:
        """Return a unique fingerprint for the dataset provider."""

        return sha1(f"{self.dataset}-{self.subset}-{self.tokenizer}".encode("ascii")).hexdigest()

    def _encode_dataset(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        dtype = np.uint16 if tokenizer.vocab_size < 64 * 1024 else np.int32

        # Ensures that the loaded dataset is always a dictionary
        logger.info(f"Loading dataset: {self.dataset}/{self.subset}")
        raw_dataset = hf_load_dataset(self.dataset, self.subset)
        if not isinstance(raw_dataset, DatasetDict):
            raw_dataset = DatasetDict({"train": raw_dataset})

        # Ensures that `validation` and `test` splits are present
        if "validation" not in raw_dataset:
            logger.info("Creating validation split ...")

            tmp_dataset_dict = raw_dataset["train"].train_test_split(
                test_size=self.validation_split, shuffle=True, seed=self.seed
            )
            raw_dataset["train"] = tmp_dataset_dict["train"]
            raw_dataset["validation"] = tmp_dataset_dict["test"]

        if "test" not in raw_dataset:
            logger.info("Creating test split ...")

            tmp_dataset_dict = raw_dataset["validation"].train_test_split(test_size=0.25, shuffle=True, seed=self.seed)
            raw_dataset["validation"] = tmp_dataset_dict["train"]
            raw_dataset["test"] = tmp_dataset_dict["test"]

        logger.info("Encoding dataset ...")
        column_names = raw_dataset["train"].column_names
        encoded_dataset = raw_dataset.map(
            tokenize_concatenated_dataset,
            fn_kwargs={
                "tokenizer": tokenizer,
                "mapping_column_name": self.mapping_column_name,
                "dtype": dtype,
            },
            batched=True,
            num_proc=self.num_workers,
            remove_columns=column_names,
        )

        if self.use_shared_memory:
            dataset_dict = process_with_shared_memory(encoded_dataset, dtype, num_proc=self.num_workers)
        else:
            dataset_dict = process_with_memory_map_files(
                encoded_dataset, self.cache_dir, dtype, num_proc=self.num_workers
            )

        logger.info(f"Saving dataset to: {self.cache_dir}")
        for split, dataset in dataset_dict.items():
            np.save(self.cache_dir / f"{split}.npy", dataset)

            # If using shared memory, dataset needs to have its shared memory
            # unlinked to prevent memory leak
            if self.use_shared_memory:
                dataset.shm.unlink()

            # If not using shared memory, dataset needs to have its memory map
            # closed to prevent an additional .bin file
            if not self.use_shared_memory:
                dataset._mmap.close()
                Path(self.cache_dir / f"{split}.bin").unlink()

    @overrides
    def get_train_dataset(self, seq_len: Optional[int] = 1) -> FastHfDataset:
        assert self.cache_dir.is_dir()
        input_ids = np.load(self.cache_dir / "train.npy", mmap_mode="r")

        return FastHfDataset(input_ids, seq_len=seq_len)

    @overrides
    def get_val_dataset(self, seq_len: Optional[int] = 1) -> FastHfDataset:
        assert self.cache_dir.is_dir()
        input_ids = np.load(self.cache_dir / "validation.npy", mmap_mode="r")

        return FastHfDataset(input_ids, seq_len=seq_len)

    @overrides
    def get_test_dataset(self, seq_len: Optional[int] = 1) -> FastHfDataset:
        assert self.cache_dir.is_dir()
        input_ids = np.load(self.cache_dir / "test.npy", mmap_mode="r")

        return FastHfDataset(input_ids, seq_len=seq_len)
