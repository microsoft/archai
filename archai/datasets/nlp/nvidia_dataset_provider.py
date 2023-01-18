# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Optional

from overrides import overrides

from archai.api.dataset_provider import DatasetProvider
from archai.common.distributed_utils import sync_workers
from archai.datasets.nlp.nvidia_dataset_provider_utils import Corpus


class NvidiaDatasetProvider(DatasetProvider):
    """NVIDIA dataset provider."""

    def __init__(
        self,
        dataset_name: Optional[str] = "wt103",
        dataset_dir: Optional[str] = "",
        cache_dir: Optional[str] = "cache",
        vocab_type: Optional[str] = "gpt2",
        vocab_size: Optional[int] = None,
        refresh_cache: Optional[bool] = False,
    ) -> None:
        """Initializes NVIDIA dataset provider.

        Args:
            dataset_name: Name of the dataset.
            dataset_dir: Dataset folder.
            cache_dir: Path to the cache folder.
            vocab_type: Type of vocabulary/tokenizer.
            vocab_size: Vocabulary size.
            refresh_cache: Whether cache should be refreshed.

        """

        super().__init__()

        self.corpus = Corpus(
            dataset_name, dataset_dir, cache_dir, vocab_type, vocab_size=vocab_size, refresh_cache=refresh_cache
        )

        if not self.corpus.load():
            self.corpus.train_and_encode()

            with sync_workers() as rank:
                if rank == 0 and dataset_name != "lm1b":
                    self.corpus.save_cache()

    @overrides
    def get_train_dataset(self) -> List[int]:
        return self.corpus.train

    @overrides
    def get_val_dataset(self) -> List[int]:
        return self.corpus.valid

    @overrides
    def get_test_dataset(self) -> List[int]:
        return self.corpus.test
