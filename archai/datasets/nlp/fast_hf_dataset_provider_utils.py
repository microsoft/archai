# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Copyright (c) Hazy Research.
# Licensed under the BSD-3-Clause license.
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules

from __future__ import annotations

import math
import mmap
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset

# `multiprocessing.shared_memory` is only available in Python 3.8+`
if sys.version_info.major == 3 and sys.version_info.minor >= 8:
    from multiprocessing.shared_memory import SharedMemory


class FastHfDataset(Dataset):
    """Fast Hugging Face dataset."""

    def __init__(self, input_ids: torch.Tensor, seq_len: Optional[int] = 1) -> None:
        """Initialize the dataset.

        Args:
            input_ids: Tensor with the inputs (encoded data).
            seq_len: Sequence length.

        """

        super().__init__()

        self.n_input_ids = ((len(input_ids) - 1) // seq_len) * seq_len + 1
        self.seq_len = seq_len

        # `input_ids` should not be sliced since they could be memory mapped
        self.input_ids = input_ids
        self.n_sequences = math.ceil((self.n_input_ids - 1) / self.seq_len)

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = idx * self.seq_len
        seq_len = min(self.seq_len, self.n_input_ids - 1 - start_idx)

        input_ids = torch.as_tensor(self.input_ids[start_idx : (start_idx + seq_len + 1)].astype(np.int64))
        labels = input_ids[1:].clone()

        return input_ids[:-1], labels


class SHMArray(np.ndarray):
    """Numpy array compatible with SharedMemory from `multiprocessing.shared_memory`.

    Reference:
        https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array

    """

    def __new__(cls: SHMArray, input_array: np.ndarray, shm: Optional[SharedMemory] = None) -> SHMArray:
        obj = np.asarray(input_array).view(cls)
        obj.shm = shm

        return obj

    def __array_finalize__(self, obj: SHMArray) -> None:
        if obj is None:
            return

        self.shm = getattr(obj, "shm", None)


def process_with_shared_memory(
    dataset_dict: DatasetDict, dtype: np.dtype, num_proc: Optional[int] = 1
) -> Dict[str, SHMArray]:
    """Process the dataset with a shared memory.

    Args:
        dataset_dict: Dataset dictionary.
        dtype: Numpy data type.
        num_proc: Number of processes.

    Returns:
        Dictionary with shared memory-processed datasets.

    """

    def _process_with_shared_memory(example: Dict[str, Any], name, length: int) -> None:
        shared_memory = SharedMemory(name=name)

        shared_memory_array = np.ndarray((length,), dtype=dtype, buffer=shared_memory.buf)
        start_idx = example["offset"] - len(example["input_ids"])
        shared_memory_array[start_idx : example["offset"]] = example["input_ids"]

        shared_memory.close()

    processed_dataset_dict = {}
    for name, ds in dataset_dict.items():
        dataset_dict[name] = ds.add_column("offset", np.cumsum(ds["length"]))
        length = dataset_dict[name][-1]["offset"]

        shared_memory = SharedMemory(create=True, size=length * np.dtype(dtype).itemsize)
        shared_memory_name = shared_memory.name

        dataset_dict[name].map(
            _process_with_shared_memory,
            fn_kwargs={"name": shared_memory_name, "length": length},
            batched=False,
            num_proc=num_proc,
        )

        shared_memory_array = np.ndarray((length,), dtype=dtype, buffer=shared_memory.buf)
        processed_dataset_dict[name] = SHMArray(shared_memory_array, shm=shared_memory)

    return processed_dataset_dict


def process_with_memory_map_files(
    dataset_dict: DatasetDict, cache_dir: str, dtype: np.dtype, num_proc: Optional[int] = 1
) -> Dict[str, np.ndarray]:
    """Process the dataset with memory map files.

    Args:
        dataset_dict: Dataset dictionary.
        cache_dir: Cache directory.
        dtype: Numpy data type.
        num_proc: Number of processes.

    Returns:
        Dictionary with memory map file-processed datasets.

    """

    def _process_with_memory_map_files(example: Dict[str, Any], file_path: str) -> None:
        with open(file_path, "r+b") as f:
            memory_map = mmap.mmap(f.fileno(), 0)

            start_idx = example["offset"] - len(example["input_ids"])
            length = len(example["input_ids"])

            memory_map_array = np.ndarray(
                (length,), dtype=dtype, buffer=memory_map, offset=np.dtype(dtype).itemsize * start_idx
            )
            memory_map_array[:] = example["input_ids"]

            memory_map.flush()

    processed_dataset_dict = {}
    for split, dataset in dataset_dict.items():
        dataset_dict[split] = dataset.add_column("offset", np.cumsum(dataset["length"]))
        length = dataset_dict[split][-1]["offset"]

        file_path = cache_dir / f"{split}.bin"
        with open(file_path.as_posix(), "wb") as f:
            f.truncate(length * np.dtype(dtype).itemsize)

        dataset_dict[split].map(
            _process_with_memory_map_files,
            fn_kwargs={"file_path": file_path},
            batched=False,
            num_proc=num_proc,
        )

        processed_dataset_dict[split] = np.memmap(file_path, dtype=dtype, mode="r", shape=(length,))

    return processed_dataset_dict
