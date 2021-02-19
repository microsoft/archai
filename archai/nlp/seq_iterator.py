

from typing import List

import numpy as np

from archai.common.apex_utils import ApexUtils
from archai.common.common import logger
from archai.common import utils, apex_utils
from archai.common.config import Config

class SeqIterator:
    def __init__(self, conf_loader:Config, tokens:List[int], batch_size:int, n_seg:int, n_seq:int):
        conf_apex  = conf_loader['apex']
        apex = apex_utils.ApexUtils(conf_apex, logger)

        self.tokens = tokens
        self.batch_size = batch_size
        self.n_seq = n_seq
        self.n_seg = n_seg

        assert n_seg % n_seq == 0
        n = len(self.tokens) - 1

        start, end = 0, n
        if apex.is_dist():
            start = n * apex.local_rank // apex.world_size
            end = n * (apex.local_rank + 1) // apex.world_size

        if batch_size > 1:
            span_i = (end - start) // batch_size // n_seg * n_seg
            span = span_i * batch_size
            end = start + span
        else:
            span_i = end - start
        self.span_i = span_i
        self.starts = np.arange(start, end, span_i)

    def __iter__(self):
        tokens = self.tokens
        starts = self.starts

        for i in range(0, self.span_i, self.n_seg):
            starts_i = starts + i
            batch_i = np.array([tokens[s_i: s_i + self.n_seg + 1] for s_i in starts_i])
            yield batch_i.astype(np.int64)

    def __len__(self):
        return len(range(0, self.span_i, self.n_seg))