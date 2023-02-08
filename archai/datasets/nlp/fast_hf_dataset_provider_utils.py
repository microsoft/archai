# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Copyright (c) Hazy Research.
# Licensed under the BSD-3-Clause license.
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules

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
