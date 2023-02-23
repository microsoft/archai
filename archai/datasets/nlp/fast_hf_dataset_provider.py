# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from overrides import overrides
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from archai.api.dataset_provider import DatasetProvider
from archai.common.ordered_dict_logger import OrderedDictLogger
from archai.datasets.nlp.fast_hf_dataset_provider_utils import (
    FastHfDataset,
    SHMArray,
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
        train_file: str,
        validation_file: str,
        test_file: str,
        tokenizer: Optional[AutoTokenizer] = None,
    ) -> None:
        """Initialize Fast Hugging Face-based dataset provider.

        Args:
            train_file: Path to the training array file (.npy).
            validation_file: Path to the validation array file (.npy).
            test_file: Path to the test array file (.npy).
            tokenizer: Instance of tokenizer to use.

        """

        super().__init__()

        self.train_file = train_file
        self.validation_file = validation_file
        self.test_file = test_file
        self.tokenizer = tokenizer

    @staticmethod
    def _encode_dataset(
        dataset_dict: DatasetDict,
        tokenizer: AutoTokenizer,
        mapping_column_name: List[str],
        use_eos_token: bool,
        dtype: np.dtype,
        num_workers: int,
    ) -> DatasetDict:
        logger.info("Encoding dataset ...")
        logger.info(f"Number of workers: {num_workers} | EOS token: {use_eos_token}")

        column_names = dataset_dict["train"].column_names
        encoded_dataset_dict = dataset_dict.map(
            tokenize_concatenated_dataset,
            fn_kwargs={
                "tokenizer": tokenizer,
                "mapping_column_name": mapping_column_name,
                "use_eos_token": use_eos_token,
                "dtype": dtype,
            },
            batched=True,
            num_proc=num_workers,
            remove_columns=column_names,
        )

        return encoded_dataset_dict

    @staticmethod
    def _process_dataset_to_memory(
        dataset_dict: DatasetDict, cache_dir: str, dtype: np.dtype, num_workers: int, use_shared_memory: int
    ) -> Dict[str, Union[SHMArray, np.ndarray]]:
        logger.info("Processing dataset to memory ...")
        logger.info(f"Number of workers: {num_workers} | Shared memory: {use_shared_memory}")

        if use_shared_memory:
            return process_with_shared_memory(dataset_dict, dtype, num_proc=num_workers)

        return process_with_memory_map_files(dataset_dict, cache_dir, dtype, num_proc=num_workers)

    @staticmethod
    def _save_dataset(
        dataset_dict: Dict[str, Union[SHMArray, np.ndarray]],
        tokenizer: AutoTokenizer,
        cache_dir: str,
        use_shared_memory: bool,
    ) -> Tuple[Path, Path, Path]:
        logger.info(f"Saving dataset to: {cache_dir}")

        cache_files = ()
        for split, dataset in dataset_dict.items():
            np.save(cache_dir / f"{split}.npy", dataset)

            # If using shared memory, dataset needs to have its shared memory
            # unlinked to prevent memory leak
            if use_shared_memory:
                dataset.shm.unlink()

            # If not using shared memory, dataset needs to have its memory map
            # closed to prevent an additional .bin file
            if not use_shared_memory:
                dataset._mmap.close()
                Path(cache_dir / f"{split}.bin").unlink()

            cache_files += (cache_dir / f"{split}.npy",)

        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)

        return cache_files

    @classmethod
    def from_hub(
        cls: FastHfDatasetProvider,
        dataset_name: str,
        dataset_config_name: Optional[str] = None,
        data_dir: Optional[str] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        tokenizer_name: Optional[str] = None,
        tokenizer_max_length: Optional[int] = None,
        mapping_column_name: Optional[List[str]] = None,
        validation_split: Optional[float] = 0.0,
        seed: Optional[int] = 42,
        num_workers: Optional[int] = 1,
        use_eos_token: Optional[bool] = True,
        use_shared_memory: Optional[bool] = True,
        cache_dir: Optional[str] = "cache",
    ) -> FastHfDatasetProvider:
        """Load a dataset provider by downloading and encoding data from Hugging Face Hub.

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

        Returns:
            Dataset provider.

        """

        assert xor(tokenizer, tokenizer_name), "`tokenizer` and `tokenizer_name` are mutually exclusive."
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer_max_length:
            logger.warn(f"New maximum length set for the tokenizer: {tokenizer_max_length}.")
            tokenizer.model_max_length = tokenizer_max_length

        dtype = np.uint16 if tokenizer.vocab_size < 64 * 1024 else np.int32
        use_shared_memory = use_shared_memory and ALLOW_SHARED_MEMORY

        cache_dir = Path(cache_dir)
        if cache_dir.is_dir():
            logger.warn(f"Cache: {cache_dir} already exists and will be overritten.")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Ensure that the downloaded dataset is always a dictionary
        logger.info("Downloading dataset ...")
        hub_dataset_dict = load_dataset(dataset_name, name=dataset_config_name, data_dir=data_dir)
        if not isinstance(hub_dataset_dict, DatasetDict):
            hub_dataset_dict = DatasetDict({"train": hub_dataset_dict})

        # Ensure that `validation` and `test` splits are available
        if "validation" not in hub_dataset_dict:
            logger.info("Creating validation split ...")

            validation_split = validation_split or 0.1
            tmp_dataset_dict = hub_dataset_dict["train"].train_test_split(
                test_size=validation_split, shuffle=True, seed=seed
            )
            hub_dataset_dict["train"] = tmp_dataset_dict["train"]
            hub_dataset_dict["validation"] = tmp_dataset_dict["test"]

        if "test" not in hub_dataset_dict:
            logger.info("Creating test split ...")

            tmp_dataset_dict = hub_dataset_dict["validation"].train_test_split(test_size=0.25, shuffle=True, seed=seed)
            hub_dataset_dict["validation"] = tmp_dataset_dict["train"]
            hub_dataset_dict["test"] = tmp_dataset_dict["test"]

        encoded_dataset_dict = FastHfDatasetProvider._encode_dataset(
            hub_dataset_dict, tokenizer, mapping_column_name, use_eos_token, dtype, num_workers
        )
        processed_dataset_dict = FastHfDatasetProvider._process_dataset_to_memory(
            encoded_dataset_dict, cache_dir, dtype, num_workers, use_shared_memory
        )

        cache_files = FastHfDatasetProvider._save_dataset(
            processed_dataset_dict, tokenizer, cache_dir, use_shared_memory
        )

        return FastHfDatasetProvider(*cache_files, tokenizer=tokenizer)

    @classmethod
    def from_cache(cls: FastHfDatasetProvider, cache_dir: str) -> FastHfDatasetProvider:
        """Load a dataset provider from a cache directory.

        Args:
            cache_dir: Path to the cache directory.

        Returns:
            Dataset provider.

        """

        logger.info(f"Loading dataset from: {cache_dir}")

        cache_dir = Path(cache_dir)
        cache_train_file = cache_dir / "train.npy"
        cache_validation_file = cache_dir / "validation.npy"
        cache_test_file = cache_dir / "test.npy"

        tokenizer_file = cache_dir / "tokenizer.pkl"
        if not tokenizer_file.is_file():
            logger.warn(f"Could not find tokenizer in {cache_dir}.")
            tokenizer = None
        else:
            with open(tokenizer_file, "rb") as f:
                tokenizer = pickle.load(f)

        return FastHfDatasetProvider(cache_train_file, cache_validation_file, cache_test_file, tokenizer=tokenizer)

    @overrides
    def get_train_dataset(self, seq_len: Optional[int] = 1) -> FastHfDataset:
        input_ids = np.load(self.train_file, mmap_mode="r")

        return FastHfDataset(input_ids, seq_len=seq_len)

    @overrides
    def get_val_dataset(self, seq_len: Optional[int] = 1) -> FastHfDataset:
        input_ids = np.load(self.validation_file, mmap_mode="r")

        return FastHfDataset(input_ids, seq_len=seq_len)

    @overrides
    def get_test_dataset(self, seq_len: Optional[int] = 1) -> FastHfDataset:
        input_ids = np.load(self.test_file, mmap_mode="r")

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
