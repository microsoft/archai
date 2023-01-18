# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Optional, Union

from overrides import overrides

from archai.api.data_loader_provider import DataLoaderProvider
from archai.datasets.nlp.nvidia_data_loader_provider_utils import (
    LMMultiFileIterator,
    LMOrderedIterator,
)


class NvidiaDataLoaderProvider(DataLoaderProvider):
    """NVIDIA data loader provider."""

    def __init__(self, dataset_name: Optional[str] = "wt103") -> None:
        """Initializes NVIDIA data loader provider.

        Args:
            dataset_name: Name of the dataset.

        """

        super().__init__()

        self.dataset_name = dataset_name

    @overrides
    def get_data_loader(
        self,
        input_ids: List[int],
        batch_size: int,
        seq_len: int,
        device: Optional[str] = "cpu",
        mem_len: Optional[int] = 0,
        ext_len: Optional[int] = 0,
    ) -> Union[LMOrderedIterator, LMMultiFileIterator]:
        if self.dataset_name in ["wt2", "wt103"] or self.dataset_name.startswith("olx_"):
            return LMOrderedIterator(input_ids, batch_size, seq_len, device=device, mem_len=mem_len, ext_len=ext_len)

        elif self.dataset_name == "lm1b":
            return LMMultiFileIterator(
                input_ids,
                self.vocab,
                batch_size,
                seq_len,
                device=device,
                mem_len=mem_len,
                ext_len=ext_len,
            )

        else:
            raise RuntimeError(f"Dataset: {self.dataset_name} is not supported yet.")
