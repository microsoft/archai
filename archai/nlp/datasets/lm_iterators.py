# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Language Modeling iterators.
"""

from typing import List, Optional, Tuple
import numpy as np
import torch
from archai.nlp.datasets.distributed_utils import distributed
from archai.nlp.datasets.tokenizer_utils.vocab_base import VocabBase


class LMOrderedIterator:
    """Language Modeling ordered iterator.

    """

    def __init__(self,
                 input_ids: torch.LongTensor,
                 bsz: int,
                 bptt: int,
                 device: Optional[str] = 'cpu',
                 mem_len: Optional[int] = None,
                 ext_len: Optional[int] = None,
                 warmup: Optional[bool] = True) -> None:
        """Overrides initialization method.

        Args:
            input_ids: Input identifiers.
            bsz: Batch size.
            bptt: Number of backpropagations through time.
            device: Deviced to be used.
            mem_len: Length of the memory.
            ext_len: Length of the extended context.
            warmup: Whether to supply warmup batches or not.

        """

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0
        self.mem_len = mem_len
        self.warmup = warmup

        self.device = device

        # Work out how cleanly we can divide the dataset into bsz parts.
        n_step = input_ids.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        input_ids = input_ids[:n_step * bsz]

        # Evenly divide the input_ids across the bsz batches.
        self.input_ids = input_ids.view(bsz, -1).contiguous().pin_memory()

        if mem_len and warmup:
            self.warmup_batches = (mem_len + bptt - 1) // bptt
            self.warmup_elems = self.warmup_batches * bptt

            warmup_ids = self.input_ids.roll((self.warmup_elems, 1), (1, 0))[:, :self.warmup_elems]
            self.input_ids = torch.cat((warmup_ids, self.input_ids), dim=-1)

        # Partition input_ids for DistributedDataParallel
        world_size = distributed.get_world_size()
        rank = distributed.get_rank()
        self.input_ids = self.input_ids.chunk(world_size, dim=0)[rank]

        # Number of mini-batches
        self.n_batch = (self.input_ids.size(1) + self.bptt - 1) // self.bptt

        self.last_iter = None

    def roll(self, seed: int) -> None:
        """Unrolls a continuous tensor into batches.

        Args:
            seed: Random seed.

        """

        rng = torch.Generator()
        rng.manual_seed(seed)
        for i in range(self.input_ids.size(0)):
            row = self.input_ids[i, :]
            shift = torch.randint(0, self.input_ids.size(1), (1,), generator=rng)
            row = torch.cat((row[shift:], row[:shift]))
            self.input_ids[i, :] = row

    def get_batch(self,
                  i: int,
                  bptt: Optional[int] = None) -> Tuple[torch.LongTensor, torch.LongTensor, int, bool]:
        """Gathers batch based on supplied identifier.

        Args:
            i: Batch identifier.
            bptt: Number of backpropagations through time.

        Returns:
            (Tuple[torch.LongTensor, torch.LongTensor, int, bool]): Inputs tensor, labels tensor,
                length of sequence and whether warmup has been applied or not.

        """

        if bptt is None:
            bptt = self.bptt

        seq_len = min(bptt, self.input_ids.size(1) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        input_ids = self.input_ids[:,beg_idx:end_idx].to(self.device, non_blocking=True)
        labels = self.input_ids[:,i+1:i+1+seq_len].to(self.device, non_blocking=True)

        if self.mem_len and self.warmup:
            warm = i >= self.warmup_elems
        else:
            warm = True

        return input_ids, labels, seq_len, warm

    def get_fixlen_iter(self,
                        start: Optional[int] = 0) -> Tuple[torch.LongTensor, torch.LongTensor, int, bool]:
        """Gets a fixed length iterator.

        Args:
            start: Starting batch identifier.

        Yields:
            (Tuple[torch.LongTensor, torch.LongTensor, int, bool]): Inputs tensor, labels tensor,
                length of sequence and whether warmup has been applied or not.

        """

        if start != 0:
            start += self.bptt

        for i in range(start, self.input_ids.size(1) - 1, self.bptt):
            self.last_iter = i

            yield self.get_batch(i)

    def get_varlen_iter(self,
                        start: Optional[int] = 0,
                        std: Optional[int] = 5,
                        min_len: Optional[int] = 5,
                        max_deviation: Optional[int] = 3) -> (Tuple[torch.LongTensor, torch.LongTensor, int]):
        """Gets a variable length iterator.

        Args:
            start: Starting batch identifier.
            std: Standard deviation.
            min_len: Minimum length of sequence.
            max_deviation: Maximum deviation of sequence length.

        Yields:
            (Tuple[torch.LongTensor, torch.LongTensor, int]): Inputs tensor, labels tensor,
                and length of sequence.

        """

        max_len = self.bptt + max_deviation * std
        i = start

        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))

            input_ids, labels, seq_len = self.get_batch(i, bptt)

            i += seq_len

            yield input_ids, labels, seq_len

            if i >= self.input_ids.size(1) - 2:
                break

    def __iter__(self) -> callable:
        """Gathers the iterator.

        Returns:
            (callable): Callable containing the iterator.

        """

        return self.get_fixlen_iter()


class LMShuffledIterator:
    """Language Modeling shuffled iterator.

    """

    def __init__(self,
                 input_ids: torch.LongTensor,
                 bsz: int,
                 bptt: int,
                 device: Optional[str] = 'cpu',
                 ext_len: Optional[int] = None,
                 shuffle: Optional[bool] = False) -> None:
        """Overrides initialization method.

        Args:
            input_ids: Input identifiers.
            bsz: Batch size.
            bptt: Number of backpropagations through time.
            device: Deviced to be used.
            ext_len: Length of the extended context.
            shuffle: Whether to shuffle batches or not.

        """

        self.input_ids = input_ids

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self) -> List[int]:
        """Gets a stream of sentences.

        Yields:
            (List[int]): Encoced sentence.

        """
        
        # index iterator
        epoch_indices = np.random.permutation(len(self.input_ids)) if self.shuffle \
            else np.array(range(len(self.input_ids)))

        # sentence iterator
        for idx in epoch_indices:
            yield self.input_ids[idx]

    def stream_iterator(self,
                        sent_stream: List[int]) -> Tuple[torch.LongTensor, torch.LongTensor, int]:
        """Iterates through a stream of sentences.

        Args:
            sent_stream: Sentence stream.

        Yields:
            (Tuple[torch.LongTensor, torch.LongTensor, int]): Inputs tensor, labels tensor and
                number of backpropagations through time.

        """

        # streams for each input_ids in the batch
        streams = [None] * self.bsz

        input_ids = torch.LongTensor(self.bsz, self.bptt)
        labels = torch.LongTensor(self.bsz, self.bptt)

        n_retain = 0

        while True:
            # input_ids   : [bsz x n_retain+bptt]
            # labels : [bsz x bptt]
            input_ids[:, n_retain:].fill_(-1)
            labels.fill_(-1)

            valid_batch = True

            for i in range(self.bsz):
                n_filled = 0

                try:
                    while n_filled < self.bptt:
                        if streams[i] is None or len(streams[i]) <= 1:
                            streams[i] = next(sent_stream)

                        # number of new tokens to fill in
                        n_new = min(len(streams[i]) - 1, self.bptt - n_filled)

                        # first n_retain tokens are retained from last batch
                        input_ids[i, n_retain+n_filled:n_retain+n_filled+n_new] = \
                            streams[i][:n_new]
                        labels[i, n_filled:n_filled+n_new] = \
                            streams[i][1:n_new+1]

                        streams[i] = streams[i][n_new:]
                        n_filled += n_new

                except StopIteration:
                    valid_batch = False

                    break

            if not valid_batch:
                return

            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            yield input_ids, labels, self.bptt

            n_retain = min(input_ids.size(1), self.ext_len)
            if n_retain > 0:
                input_ids[:, :n_retain] = input_ids[:, -n_retain:]

            input_ids.resize_(input_ids.size(0), n_retain + self.bptt)

    def __iter__(self) -> Tuple[torch.LongTensor, torch.LongTensor, int]:
        """Gathers the iterator.

        Yields:
            (Tuple[torch.LongTensor, torch.LongTensor, int]): Inputs tensor, labels tensor and
                number of backpropagations through time.

        """

        # sent_stream is an iterator
        sent_stream = self.get_sent_stream()

        for batch in self.stream_iterator(sent_stream):
            yield batch


class LMMultiFileIterator(LMShuffledIterator):
    """Language Modeling multi-file iterator.

    """

    def __init__(self,
                 paths: List[str],
                 vocab: VocabBase,
                 bsz: int,
                 bptt: int,
                 device: Optional[str] = 'cpu',
                 ext_len: Optional[int] = None,
                 shuffle: Optional[bool] = False) -> None:
        """Overrides initialization method.

        Args:
            paths: Input files.
            vocab: Vocabulary.
            bsz: Batch size.
            bptt: Number of backpropagations through time.
            device: Deviced to be used.
            ext_len: Length of the extended context.
            shuffle: Whether to shuffle batches or not.

        """

        self.paths = paths
        self.vocab = vocab

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self, path: str) -> List[str]:
        """Gets a stream of sentences.

        Args:
            path: File path.

        Returns:
            (List[str]): Stream of sentences.

        """

        sents = self.vocab.encode_file(path, add_double_eos=True)

        if self.shuffle:
            np.random.shuffle(sents)

        sent_stream = iter(sents)

        return sent_stream

    def __iter__(self) -> Tuple[torch.LongTensor, torch.LongTensor, int]:
        """Gathers the iterator.

        Yields:
            (Tuple[torch.LongTensor, torch.LongTensor, int]): Inputs tensor, labels tensor and
                number of backpropagations through time.

        """
        
        if self.shuffle:
            np.random.shuffle(self.paths)

        for path in self.paths:
            # sent_stream is an iterator
            sent_stream = self.get_sent_stream(path)

            for batch in self.stream_iterator(sent_stream):
                yield batch
