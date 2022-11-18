# Copyright (c) 2019-2020, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0.
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/data_utils.py

"""Language Modeling-based iterators.
"""

from typing import Generator, Iterator, List, Optional, Tuple

import numpy as np
import torch

from archai.nlp.datasets.nvidia import distributed_utils
from archai.nlp.datasets.nvidia.tokenizer_utils.vocab_base import VocabBase


class LMOrderedIterator:
    """Implements an ordered-token iterator, e.g., there is no padding as tokens are contiguous."""

    def __init__(
        self,
        input_ids: torch.LongTensor,
        bsz: int,
        bptt: int,
        device: Optional[str] = "cpu",
        mem_len: Optional[int] = 0,
        ext_len: Optional[int] = 0,
        warmup: Optional[bool] = True,
    ) -> None:
        """Initializes by sharding inputs across GPUs, if distributed training is available.

        Args:
            input_ids: Inputs.
            bsz: Batch size.
            bptt: Sequence length (backpropagation through time).
            device: Device to place the iterator.
            mem_len: Length of memory (for Transformer-XL).
            ext_len: Length of extended context (for Transformer-XL).
            warmup: Whether warmup batches should be created.

        """

        self.bsz = bsz
        self.bptt = bptt
        self.device = device
        self.ext_len = ext_len
        self.mem_len = mem_len
        self.warmup = warmup
        self.last_iter = None

        # Divides cleanly the inputs into batches and trims the remaining elements
        n_step = input_ids.size(0) // bsz
        input_ids = input_ids[: n_step * bsz]
        self.input_ids = input_ids.view(bsz, -1).contiguous().pin_memory()

        # Creates warmup batches if memory is being used
        if mem_len and warmup:
            self.warmup_batches = (mem_len + bptt - 1) // bptt
            self.warmup_elems = self.warmup_batches * bptt

            warmup_ids = self.input_ids.roll((self.warmup_elems, 1), (1, 0))[:, : self.warmup_elems]
            self.input_ids = torch.cat((warmup_ids, self.input_ids), dim=-1)

        # Chunks the inputs for distributed training (if available)
        world_size = distributed_utils.get_world_size()
        rank = distributed_utils.get_rank()
        self.input_ids = self.input_ids.chunk(world_size, dim=0)[rank]

        self.n_batch = (self.input_ids.size(1) + self.bptt - 1) // self.bptt

    def roll(self, seed: int) -> None:
        """Rolls/shifts the data according to a random seed.

        Args:
            seed: Seed used to roll/shift the data.

        """

        rng = torch.Generator()
        rng.manual_seed(seed)

        for i in range(self.input_ids.size(0)):
            shift = torch.randint(0, self.input_ids.size(1), (1,), generator=rng)

            row = self.input_ids[i, :]
            row = torch.cat((row[shift:], row[:shift]))

            self.input_ids[i, :] = row

    def get_batch(self, i: int, bptt: Optional[int] = None) -> Tuple[torch.LongTensor, torch.LongTensor, int, bool]:
        """Gets a batch of `bptt` size.

        Args:
            i: Identifier of batch.
            bptt: Sequence length.

        Returns:
            (Tuple[torch.LongTensor, torch.LongTensor, int, bool]): Inputs, labels,
                sequence length and whether it is a warmup batch.

        """

        if bptt is None:
            bptt = self.bptt

        seq_len = min(bptt, self.input_ids.size(1) - 1 - i)

        start_idx = max(0, i - self.ext_len)
        end_idx = i + seq_len

        input_ids = self.input_ids[:, start_idx:end_idx].to(self.device, non_blocking=True)
        labels = self.input_ids[:, i + 1 : i + 1 + seq_len].to(self.device, non_blocking=True)

        warmup = True
        if self.mem_len and self.warmup:
            warmup = i >= self.warmup_elems

        return input_ids, labels, seq_len, warmup

    def get_fixlen_iter(self, start: Optional[int] = 0) -> Generator[Tuple, None, None]:
        """Gets a fixed-length iterator.

        Args:
            start: Starting point.

        Yields:
            (Generator[Tuple, None, None]): Fixed-length batches.

        """

        if start != 0:
            start += self.bptt

        for i in range(start, self.input_ids.size(1) - 1, self.bptt):
            self.last_iter = i
            yield self.get_batch(i)

    def get_varlen_iter(
        self,
        start: Optional[int] = 0,
        std: Optional[float] = 5.0,
        min_len: Optional[int] = 5,
        max_std: Optional[float] = 3.0,
    ) -> Generator[Tuple, None, None]:
        """Gets a variable-length iterator.

        Args:
            start: Starting point.
            std: Standard deviation.
            min_len: Minimum length.
            max_std: Max standard deviation.

        Yields:
            (Generator[Tuple, None, None]): Variable-length batches.

        """

        max_len = self.bptt + max_std * std
        i = start

        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.0
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))

            input_ids, labels, seq_len = self.get_batch(i, bptt)
            i += seq_len

            yield input_ids, labels, seq_len
            if i >= self.input_ids.size(1) - 2:
                break

    def __iter__(self) -> Generator[Tuple, None, None]:
        """Defaults standard iteration to fixed-length batches.

        Returns:
            (Generator[Tuple, None, None]): Fixed-length batches.

        """

        return self.get_fixlen_iter()


class LMMultiFileIterator:
    """Implements a multi-file non-ordered iterator, e.g., tokens are contiguous yet they come
    from different files.

    """

    def __init__(
        self,
        paths: List[str],
        vocab: VocabBase,
        bsz: int,
        bptt: int,
        device: Optional[str] = "cpu",
        mem_len: Optional[int] = 0,
        ext_len: Optional[int] = 0,
        n_chunks: Optional[int] = 16,
        shuffle: Optional[bool] = False,
    ) -> None:
        """Initializes by adding support to multi-file inputs and sharding files
            across GPUs, if distributed training is available.

        Args:
            paths: Paths to input files.
            vocab: Vocabulary/tokenizer.
            bsz: Batch size.
            bptt: Sequence length (backpropagation through time).
            device: Device to place the iterator.
            mem_len: Length of memory (for Transformer-XL).
            ext_len: Length of extended context (for Transformer-XL).
            n_chunks: Number of chunks (to avoid out of memory).
            shuffle: Whether shuffling should be used.

        """

        self.vocab = vocab
        self.bsz = bsz
        self.bptt = bptt
        self.device = device
        self.ext_len = ext_len
        self.n_chunks = n_chunks
        self.shuffle = shuffle
        self.last_iter = None

        # For compatibility with LMOrderedIterator
        self.n_batch = -1

        # Divides self.paths into world-size chunks and picks chunk for corresponding rank
        world_size = distributed_utils.get_world_size()
        rank = distributed_utils.get_rank()

        chunk_len = len(paths) // world_size + 1  # it causes a slight imbalance
        paths_chunks = [paths[i : i + chunk_len] for i in range(0, len(paths), chunk_len)]
        self.paths = paths_chunks[rank]

    def roll(self, seed: Optional[int] = 0) -> None:
        """Backward compatibility for using same APIs."""

        return None

    def get_sequences(self, path: str) -> torch.LongTensor:
        """Gets sequences from a file.

        Args:
            path: Input file.

        Returns:
            (torch.LongTensor): Tensor with encoded inputs from file.

        """

        sequences = self.vocab.encode_file(path)
        if self.shuffle:
            np.random.shuffle(sequences)

        return sequences

    def stream_iterator(self, iterator: Iterator) -> Generator[Tuple, None, None]:
        """Creates a streaming-based iterator.

        Args:
            iterator: Iterator with chunks of sequences.

        Yields:
            (Generator[Tuple, None, None]): Stream-based batch.

        """

        input_ids = torch.LongTensor(self.bsz, self.bptt)
        labels = torch.LongTensor(self.bsz, self.bptt)

        n_retain = 0

        while True:
            # input_ids: [bsz x n_retain+bptt]
            # labels: [bsz x bptt]
            input_ids[:, n_retain:].fill_(-1)
            labels.fill_(-1)

            valid_batch = True
            for i in range(self.bsz):
                n_filled = 0
                try:
                    while n_filled < self.bptt:
                        stream = torch.LongTensor([next(iterator) for _ in range(self.bptt + 1)])

                        # Number of new tokens to be filled in
                        n_tokens = min(len(stream) - 1, self.bptt - n_filled)

                        # First n_tokens are retained from last batch
                        input_ids[i, n_retain + n_filled : n_retain + n_filled + n_tokens] = stream[:n_tokens]
                        labels[i, n_filled : n_filled + n_tokens] = stream[1 : n_tokens + 1]

                        n_filled += n_tokens
                except StopIteration:
                    valid_batch = False
                    break

            if not valid_batch:
                return

            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            yield input_ids, labels, self.bptt, True

            n_retain = min(input_ids.size(1), self.ext_len)
            if n_retain > 0:
                input_ids[:, :n_retain] = input_ids[:, -n_retain:]
            input_ids.resize_(input_ids.size(0), n_retain + self.bptt)

    def __iter__(self) -> Generator[Tuple, None, None]:
        """Defaults standard iteration to stream-based batches.

        Returns:
            (Generator[Tuple, None, None]): Stream-based batches.

        """

        if self.shuffle:
            np.random.shuffle(self.paths)

        for path in self.paths:
            sequences = self.get_sequences(path)
            sequences_chunks = torch.chunk(sequences, self.n_chunks, 0)

            for i in range(self.n_chunks):
                iterator = iter(sequences_chunks[i])
                for idx, batch in enumerate(self.stream_iterator(iterator)):
                    yield batch
                    self.last_iter = idx
