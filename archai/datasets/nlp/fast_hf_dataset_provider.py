# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from hashlib import sha1
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import load_dataset as hf_load_dataset
from overrides import overrides
from transformers import AutoTokenizer

from archai.api.dataset_provider import DatasetProvider
from archai.common.ordered_dict_logger import OrderedDictLogger
from archai.datasets.nlp.fast_hf_dataset_provider_utils import FastHfDataset

logger = OrderedDictLogger(source=__name__)


class FastHfDatasetProvider(DatasetProvider):
    """Fast Hugging Face-based dataset provider."""

    def __init__(
        self,
        dataset: Optional[str] = "wikitext",
        subset: Optional[str] = "wikitext-2-raw-v1",
        tokenizer: Optional[str] = "gpt2",
        cache_dir: Optional[str] = "cache",
        use_shm: Optional[bool] = True,
    ) -> None:
        """Initialize Hugging Face Hub dataset provider.

        Args:
            dataset: Name of the dataset.
            subset: Name of the dataset configuration.
            cache_dir: Path to the read/write cache directory.

        """

        super().__init__()

        self.dataset = dataset
        self.subset = subset
        self.tokenizer = tokenizer
        self.use_shm = use_shm

        self.cache_dir = Path(cache_dir) / self.fingerprint
        if self.cache_dir.is_dir():
            self._load_from_cache()
        else:
            self._encode_dataset()
            self._save_to_cache()

    @property
    def fingerprint(self) -> str:
        return sha1(f"{self.tokenizer}-{self.dataset}-{self.subset}".encode("ascii")).hexdigest()

    def _load_from_cache(self) -> None:
        logger.info(f"Loading dataset from: {self.cache_dir}")

        self.input_ids_dict = {
            split: np.load(self.cache_dir / f"{split}.npy", mmap_mode="r") for split in ["train", "validation", "test"]
        }

    def _save_to_cache(self) -> None:
        logger.info(f"Saving dataset to: {self.cache_dir}")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for split, input_ids in self.input_ids_dict:
            np.save(self.cache_dir / f"{split}.npy", input_ids)

    def _encode_dataset(self):
        logger.info("Encoding dataset ...")

        # raw_dataset = hf_load_dataset(self.dataset, self.subset)
        # tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

        self.input_ids_dict = {}

    @overrides
    def get_train_dataset(self) -> FastHfDataset:
        return FastHfDataset(self.input_ids_dict["train"], seq_len=1)

    @overrides
    def get_val_dataset(self) -> FastHfDataset:
        return FastHfDataset(self.input_ids_dict["validation"], seq_len=1)

    @overrides
    def get_test_dataset(self) -> FastHfDataset:
        return FastHfDataset(self.input_ids_dict["test"], seq_len=1)
