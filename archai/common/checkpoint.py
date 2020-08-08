# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import UserDict
from typing import Callable, Any
import weakref
import os

import torch

from .config import Config
from . import utils

_CallbackType = Callable #[['CheckPoint', *kargs: Any, **kwargs: Any], None]
class CheckPoint(UserDict):
    """Callback based checkpoint model.

    Start new checkpoint by calling new() and save it by calling commit().
    This class is also dictionary. Items that needs be saved can be done so
    by setting key, value pairs after new(). As any dictionary key is set,
    checkpoint becomes dirty. On commit(), dictionary is saved and emptied.
    Invariant: checkpoint remains dirty until commit() is called.
    """
    def __init__(self, conf_checkpoint:Config, load_existing:bool) -> None:
        super().__init__()

        # region config vars
        self.filepath = utils.full_path(conf_checkpoint['filename'])
        self.freq = conf_checkpoint['freq']
        # endregion

        self._callbacks = []

        if load_existing:
            self.load_existing()

    def load_existing(self)->bool:
        assert self.is_empty()
        if self.filepath and os.path.exists(self.filepath):
            d = torch.load(self.filepath, map_location=torch.device('cpu'))
            self.clear()
            self.update(d)
            return True
        return False

    def new(self, *kargs, **kvargs)->None:
        self.clear()
        for func, obj in self._callbacks:
            func = func() # get actual refrence from weakref
            if obj is not None:
                obj = obj() # get actual reference from weakref
                if obj is None:
                    continue # instance is gone
                func(obj, self, *kargs, **kvargs)
            elif func is not None:
                func(self, *kargs, **kvargs)
            # else func is garbage collected

    def commit(self)->None:
        assert self.filepath and not self.is_empty()
        torch.save(self.data, self.filepath)
        # clean up after commit so we don't hold up references

    def is_empty(self)->bool:
        return len(self) == 0

    # TODO: this is no longer used, should we remove it?
    def subscribe(self, callback:_CallbackType)->None:
        obj = getattr(callback, '__self__', None)
        callback_ref = weakref.ref(callback.__func__), \
                       None if obj is None else weakref.ref(obj)
        self._callbacks.append(callback_ref)