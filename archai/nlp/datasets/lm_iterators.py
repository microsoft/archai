
import numpy as np
import torch

from archai.nlp.datasets import distributed_utils


class LMOrderedIterator(object):
    def __init__(self, input_ids, bsz, bptt, device='cpu', mem_len=None, ext_len=None, warmup=True):
        """
            input_ids -- LongTensor -- the LongTensor is strictly ordered
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
        world_size = distributed_utils.distributed.get_world_size()
        rank = distributed_utils.distributed.get_rank()
        self.input_ids = self.input_ids.chunk(world_size, dim=0)[rank]

        # Number of mini-batches
        self.n_batch = (self.input_ids.size(1) + self.bptt - 1) // self.bptt

        self.last_iter = None

    def roll(self, seed):
        rng = torch.Generator()
        rng.manual_seed(seed)
        for i in range(self.input_ids.size(0)):
            row = self.input_ids[i, :]
            shift = torch.randint(0, self.input_ids.size(1), (1,), generator=rng)
            row = torch.cat((row[shift:], row[:shift]))
            self.input_ids[i, :] = row

    def get_batch(self, i, bptt=None):
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

    def get_fixlen_iter(self, start=0):
        if start != 0:
            start += self.bptt
        for i in range(start, self.input_ids.size(1) - 1, self.bptt):
            self.last_iter = i
            yield self.get_batch(i)

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
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

    def __iter__(self):
        return self.get_fixlen_iter()


class LMShuffledIterator(object):
    def __init__(self, input_ids, bsz, bptt, device='cpu', mem_len=None, ext_len=None, shuffle=False):
        """
            input_ids -- list[LongTensor] -- there is no order among the LongTensors
        """
        self.input_ids = input_ids

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self):
        # index iterator
        epoch_indices = np.random.permutation(len(self.input_ids)) if self.shuffle \
            else np.array(range(len(self.input_ids)))

        # sentence iterator
        for idx in epoch_indices:
            yield self.input_ids[idx]

    def stream_iterator(self, sent_stream):
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
                        stream = torch.LongTensor([next(sent_stream) for _ in range(self.bptt + 1)])

                        # number of new tokens to fill in
                        n_new = min(len(stream) - 1, self.bptt - n_filled)

                        # first n_retain tokens are retained from last batch
                        input_ids[i, n_retain+n_filled:n_retain+n_filled+n_new] = stream[:n_new]
                        labels[i, n_filled:n_filled+n_new] = stream[1:n_new+1]
                        n_filled += n_new
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

    def __iter__(self):
        # sent_stream is an iterator
        sent_stream = self.get_sent_stream()

        for batch in self.stream_iterator(sent_stream):
            yield batch


class LMMultiFileIterator(LMShuffledIterator):
    def __init__(self, paths, vocab, bsz, bptt, device='cpu', mem_len=None, ext_len=None,
                 n_chunks=16, shuffle=False):
        self.vocab = vocab

        # For compatibility with LMOrderedIterator
        self.n_batch = -1
        self.last_iter = None

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle
        self.n_chunks = n_chunks

        # DDP prep: partition self.paths into world size chunks 
        # and pick chunk for this rank
        world_size = distributed_utils.distributed.get_world_size()
        rank = distributed_utils.distributed.get_rank()
        chunk_len = len(paths) // world_size + 1 # NOTE: this causes a slight imbalance!
        paths_chunks = [paths[i:i+chunk_len] for i in range(0, len(paths), chunk_len)]
        self.paths = paths_chunks[rank]

    def roll(self, seed=0):
        return

    def get_sents(self, path):
        sents = self.vocab.encode_file(path)
        if self.shuffle:
            np.random.shuffle(sents)
        return sents

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.paths)

        for path in self.paths:
            sents = self.get_sents(path)
            sents_chunks = torch.chunk(sents, self.n_chunks, 0)

            for i in range(self.n_chunks):
                sent_stream = iter(sents_chunks[i])
                for idx, batch in enumerate(self.stream_iterator(sent_stream)):
                    yield batch
                    self.last_iter = idx
