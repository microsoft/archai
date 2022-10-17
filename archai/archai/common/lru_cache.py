# Copyright (c) Microsoft Corporation


from collections import OrderedDict
from copy import deepcopy
import os
import functools

class LRUCache(OrderedDict):
    """An OrderedDict version of functools.lru_cache().
    It somewhat helps when the results are not hashable.

    Limit size, evicting the least recently looked-up key when full.
    From: https://docs.python.org/3/library/collections.html
Example use:
lru = LRUCache()
if k in lru:
    score, state = lru[k]
else:
    score, state = self.model.run(None, {'state':state, 'word_indices':token})
    lru[k] = (score, state)

    """

    def __init__(self, maxsize=128):
        super().__init__()
        self.maxsize = maxsize

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)

        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]

def deepcopy_lru_cache(maxsize=128, typed=False):
    def decorator(f):
        cached_func = functools.lru_cache(maxsize, typed)(f)
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return deepcopy(cached_func(*args, **kwargs))
        return wrapper
    return decorator