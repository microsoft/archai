# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import torch
from datasets import load_dataset as hf_load_dataset
from datasets.dataset_dict import DatasetDict
from overrides import overrides
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from archai.api.dataset_provider import DatasetProvider
from archai.common.ordered_dict_logger import OrderedDictLogger
from archai.datasets.nlp.fast_hf_dataset_provider_utils import (
    FastHfDataset,
    process_with_memory_map_files,
    process_with_shared_memory,
    xor,
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
        dataset_name: str,
        dataset_config_name: Optional[str] = None,
        data_dir: Optional[str] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        tokenizer_name: Optional[str] = None,
        tokenizer_max_length: Optional[int] = None,
        mapping_column_name: Optional[List[str]] = None,
        validation_split: Optional[float] = 0.1,
        seed: Optional[int] = 42,
        num_workers: Optional[int] = 1,
        use_eos_token: Optional[bool] = True,
        use_shared_memory: Optional[bool] = True,
        cache_dir: Optional[str] = "cache",
    ) -> None:
        """Initialize Fast Hugging Face-based dataset provider.

        The initialization consists in pre-loading the dataset, encoding it
        using the specified tokenizer, and saving it to the cache directory.

        Args:
            dataset_name: Name of the dataset.
            dataset_config_name: Name of the dataset configuration.
            data_dir: Path to the data directory.
            tokenizer: Instance of tokenizer to use.
            tokenizer_name: Name of the tokenizer, if `tokenizer` has not been passed.
            tokenizer_max_length: Maximum length of the tokenized sequences.
            mapping_column_name: The columns in `dataset` that should be tokenized.
            validation_split: Fraction of the dataset to use for validation.
            seed: Random seed.
            num_workers: Number of workers to use for encoding.
            use_eos_token: Whether to use EOS token to separate sequences.
            use_shared_memory: Whether to use shared memory for caching.
            cache_dir: Root path to the cache directory.

        """

        super().__init__()

        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.data_dir = data_dir

        assert xor(tokenizer, tokenizer_name), "`tokenizer` and `tokenizer_name` are mutually exclusive."
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer_max_length:
            logger.warn(f"New maximum length set for the tokenizer: {tokenizer_max_length}.")
            self.tokenizer.model_max_length = tokenizer_max_length

        self.mapping_column_name = mapping_column_name
        self.validation_split = validation_split
        self.seed = seed
        self.num_workers = num_workers
        self.use_eos_token = use_eos_token
        self.use_shared_memory = use_shared_memory and ALLOW_SHARED_MEMORY
        self.cache_dir = cache_dir

        # Create full path where dataset will be cached
        self.cache_data_dir = (
            Path(cache_dir)
            / self.dataset_name
            / (self.dataset_config_name or "")
            / (self.data_dir or "")
            / self.fingerprint
        )

        # If cache is not available, encode the dataset and save it to cache
        if not self.cache_data_dir.is_dir():
            self.cache_data_dir.mkdir(parents=True, exist_ok=True)
            self._encode_dataset()

    @property
    def config(self) -> Dict[str, Any]:
        """Return the configuration of the dataset provider."""

        return {
            "dataset_name": self.dataset_name,
            "dataset_config_name": self.dataset_config_name,
            "data_dir": self.data_dir,
            "tokenizer": repr(self.tokenizer),
            "mapping_column_name": self.mapping_column_name,
            "validation_split": self.validation_split,
            "seed": self.seed,
            "use_eos_token": self.use_eos_token,
            "use_shared_memory": self.use_shared_memory,
            "cache_dir": self.cache_dir,
        }

    @property
    def fingerprint(self) -> str:
        """Return a unique fingerprint for the dataset provider."""

        return sha1(repr(self.config).encode("ascii")).hexdigest()

    def _encode_dataset(self) -> None:
        dtype = np.uint16 if self.tokenizer.vocab_size < 64 * 1024 else np.int32

        # Ensure that the loaded dataset is always a dictionary
        logger.info("Downloading dataset ...")
        raw_dataset = hf_load_dataset(self.dataset_name, name=self.dataset_config_name, data_dir=self.data_dir)
        if not isinstance(raw_dataset, DatasetDict):
            raw_dataset = DatasetDict({"train": raw_dataset})

        # Ensure that `validation` and `test` splits are present
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
                "tokenizer": self.tokenizer,
                "mapping_column_name": self.mapping_column_name,
                "use_eos_token": self.use_eos_token,
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
                encoded_dataset, self.cache_data_dir, dtype, num_proc=self.num_workers
            )

        logger.info(f"Saving dataset to: {self.cache_data_dir}")
        for split, dataset in dataset_dict.items():
            np.save(self.cache_data_dir / f"{split}.npy", dataset)

            # If using shared memory, dataset needs to have its shared memory
            # unlinked to prevent memory leak
            if self.use_shared_memory:
                dataset.shm.unlink()

            # If not using shared memory, dataset needs to have its memory map
            # closed to prevent an additional .bin file
            if not self.use_shared_memory:
                dataset._mmap.close()
                Path(self.cache_data_dir / f"{split}.bin").unlink()

        with open(self.cache_data_dir / "config.json", "w") as f:
            json.dump(self.config, f)

    @classmethod
    def from_cache(cls, cache_dir: Union[str, Path]) -> FastHfDatasetProvider:
        """Load a dataset provider from a cache directory.

        Args:
            cache_dir: Path to the cache directory.

        Returns:
            Cached/encoded dataset provider.

        """

        cache_dir = Path(cache_dir)
        config_file = cache_dir / "config.json"
        if not config_file.is_file():
            raise ValueError(f"Could not find configuration in {cache_dir}.")

        with open(config_file, "r") as f:
            config = json.load(f)

        dataset_name = config.pop("dataset_name")

        return cls(dataset_name, **config)

    @overrides
    def get_train_dataset(self, seq_len: Optional[int] = 1) -> FastHfDataset:
        assert self.cache_data_dir.is_dir()
        input_ids = np.load(self.cache_data_dir / "train.npy", mmap_mode="r")

        return FastHfDataset(input_ids, seq_len=seq_len)

    @overrides
    def get_val_dataset(self, seq_len: Optional[int] = 1) -> FastHfDataset:
        assert self.cache_data_dir.is_dir()
        input_ids = np.load(self.cache_data_dir / "validation.npy", mmap_mode="r")

        return FastHfDataset(input_ids, seq_len=seq_len)

    @overrides
    def get_test_dataset(self, seq_len: Optional[int] = 1) -> FastHfDataset:
        assert self.cache_data_dir.is_dir()
        input_ids = np.load(self.cache_data_dir / "test.npy", mmap_mode="r")

        return FastHfDataset(input_ids, seq_len=seq_len)


@dataclass
class FastDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """Language modeling data collator compatible with FastHfDataset.

    Args:
        use_shifted_labels: Whether to use the original labels (shifted) or the non-shifted labels.

    """

    use_shifted_labels: bool = False

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], Mapping):
            return super().torch_call(examples)

        batch = super().torch_call([example[0] for example in examples])

        if self.use_shifted_labels:
            batch["labels"] = torch.stack([example[1] for example in examples], dim=0)

        return batch
