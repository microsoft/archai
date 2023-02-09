# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Copyright (c) Hazy Research.
# Licensed under the BSD-3-Clause license.
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules

import mmap
import subprocess
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class FastHfDataset(Dataset):
    """"""

    def __init__(self, input_ids: torch.Tensor, seq_len: Optional[int] = 1, drop_last: Optional[bool] = True) -> None:
        """"""

        super().__init__()

        if drop_last:
            self.n_input_ids = ((len(input_ids) - 1) // seq_len) * seq_len + 1
        self.seq_len = seq_len

        # `input_ids` should not be sliced since they could be memory mapped
        self.input_ids = input_ids
        self.n_sequences = np.ceil((self.n_input_ids - 1) / self.seq_len)

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = idx * self.seq_len
        seq_len = min(self.seq_len, self.n_input_ids - 1 - start_idx)

        input_ids = torch.as_tensor(self.input_ids[start_idx : (start_idx + seq_len + 1)].astype(np.int64))
        labels = input_ids[1:].clone()

        return input_ids, labels


def process_in_shared_memory():
    pass


def process_in_disk(dataset_dict, cache_dir, dtype):
    def _process_in_disk(examples, file_path):
        with open(file_path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            start_idx = examples["offset"] - len(examples["input_ids"])
            array_len = len(examples["input_ids"])
            arr = np.ndarray((array_len,), dtype=dtype, buffer=mm, offset=np.dtype(dtype).itemsize * start_idx)
            arr[:] = examples["input_ids"]
            mm.flush()

    cache_dir.mkdir(parents=True, exist_ok=True)
    input_ids_dict = {}

    for split, dataset in dataset_dict.items():
        print(split)
        dataset_dict[split] = dataset.add_column("offset", np.cumsum(dataset["length"]))
        array_len = dataset_dict[split][-1]["offset"]

        print(array_len)

        file_path = cache_dir / f"{split}.bin"
        print(array_len * np.dtype(dtype).itemsize)
        with open(file_path.as_posix(), "wb") as f:
            f.truncate(array_len * np.dtype(dtype).itemsize)
        # subprocess.run(['truncate', '-s', str(array_len * np.dtype(dtype).itemsize),
        #                 str(file_path)], check=True)

        print(file_path)

        dataset_dict[split].map(
            _process_in_disk,
            fn_kwargs={"file_path": file_path},
            batched=False,
            num_proc=1,
        )

        input_ids_dict[split] = np.memmap(file_path, dtype=dtype, mode="r", shape=(array_len,))

    print(input_ids_dict)

    return input_ids_dict
