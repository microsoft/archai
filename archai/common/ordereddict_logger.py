# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Mapping, Optional, Union, List, Iterator
from collections import OrderedDict
import logging
import time
import itertools
import yaml
import os
import shutil
import pathlib

TItems = Union[Mapping, str]

# do not reference common or otherwise we will have circular deps

def _fmt(val:Any)->str:
    if isinstance(val, float):
        return f'{val:.4g}'
    return str(val)

class OrderedDictLogger:
    """The purpose of the structured logging is to store logs as key value pair. However, when you have loop and sub routine calls, what you need is hierarchical dictionaries where the value for a key could be a dictionary. The idea is that you set one of the nodes in tree as current node and start logging your values. You can then use pushd to create and go to child node and popd to come back to parent. To implement this mechanism we use two main variables: _stack allows us to push each node on stack when pushd is called. The node is OrderedDictionary. As a convinience, we let specify child path in pushd in which case child hierarchy is created and current node will be set to the last node in specified path. When popd is called, we go back to original parent instead of parent of current node. To implement this we use _paths variable which stores subpath when each pushd call was made.
    """
    def __init__(self, filepath:Optional[str], logger:Optional[logging.Logger],
                 save_delay:Optional[float]=30.0, yaml_log=True) -> None:
        super().__init__()
        self.reset(filepath, logger, save_delay, yaml_log=yaml_log)

    def reset(self, filepath:Optional[str], logger:Optional[logging.Logger],
                 save_delay:Optional[float]=30.0,
                 load_existing_file=False, backup_existing_file=True, yaml_log=True) -> None:

        self._logger = logger
        self._yaml_log = yaml_log
        # stack stores dict for each path
        # path stores each path created via pushd
        self._paths = [['']]
        self._save_delay = save_delay
        self._call_count = 0
        self._last_save = time.time()
        self._filepath = filepath

        # backup file if already exist
        root_od = OrderedDict()
        if self._yaml_log and filepath and os.path.exists(filepath):
            if load_existing_file:
                root_od = yaml.load(self._filepath, Loader=yaml.Loader)
            if backup_existing_file:
                cur_p = pathlib.Path(filepath)
                new_p = cur_p.with_name(cur_p.stem + '.' + str(int(time.time()))
                                        + cur_p.suffix)
                if os.path.exists(str(new_p)):
                    raise RuntimeError(f'Cannot backup file {filepath} because new name {new_p} already exist')
                cur_p.rename(new_p)
        self._stack:List[Optional[OrderedDict]] = [root_od]

    def debug(self, dict:TItems, level:Optional[int]=logging.DEBUG, exists_ok=False)->None:
        self.info(dict, level, exists_ok)

    def warn(self, dict:TItems, level:Optional[int]=logging.WARN, exists_ok=False)->None:
        self.info(dict, level, exists_ok)

    def info(self, dict:TItems, level:Optional[int]=logging.INFO, exists_ok=False)->None:
        self._call_count += 1 # provides default key when key is not specified

        if isinstance(dict, Mapping): # if logging dict then just update current section
            self._update(dict, exists_ok)
            msg = ', '.join(f'{k}={_fmt(v)}' for k, v in dict.items())
        else:
            msg = dict
            key = '_warnings' if level==logging.WARN else '_messages'
            self._update_key(self._call_count, msg, node=self._root(), path=[key])

        if level is not None and self._logger:
            self._logger.log(msg=self.path() + ' ' + msg, level=level)

        if self._save_delay is not None and \
                time.time() - self._last_save > self._save_delay:
            self.save()
            self._last_save = time.time()

    def _root(self)->OrderedDict:
        r = self._stack[0]
        assert r is not None
        return r

    def _cur(self)->OrderedDict:
        self._ensure_paths()
        c = self._stack[-1]
        assert c is not None
        return c

    def save(self, filepath:Optional[str]=None)->None:
        filepath = filepath or self._filepath
        if filepath:
            with open(filepath, 'w') as f:
                yaml.dump(self._root(), f)

    def load(self, filepath:str)->None:
        with open(filepath, 'r') as f:
            od = yaml.load(f, Loader=yaml.Loader)
            self._stack = [od]

    def close(self)->None:
        self.save()
        if self._logger:
            for h in self._logger.handlers:
                h.flush()

    def _insert(self, dict:Mapping):
        self._update(dict, exists_ok=False)

    def _update(self, dict:Mapping, exists_ok=True):
        for k,v in dict.items():
            self._update_key(k, v, exists_ok)

    def _update_key(self, key:Any, val:Any, exists_ok=True,
                    node:Optional[OrderedDict]=None, path:List[str]=[]):
        if not self._yaml_log:
            return

        if not exists_ok and key in self._cur():
            raise KeyError(f'Key "{key}" already exists in log at path "{self.path()}" and cannot be updated with value {val} because it already has value "{self._cur()[key]}". Log is being saved at "{self._filepath}".')

        node = node if node is not None else self._cur()
        for p in path:
            if p not in node:
                node[p] = OrderedDict()
            node = node[p]
        node[str(key)] = val

    def _ensure_paths(self)->None:
        if not self._yaml_log:
            return

        if self._stack[-1] is not None:
            return
        last_od = None
        for i, (path, od) in enumerate(zip(self._paths, self._stack)):
            if od is None: # if corresponding dict is being delayed created
                od = last_od
                for key in path:
                    if key not in od:
                        od[key] = OrderedDict()
                    if not isinstance(od[key], OrderedDict):
                        raise RuntimeError(f'The key "{key}" is being used to store scaler value as well as in popd')
                    od = od[key]
                self._stack[i] = od
            last_od = od

    def pushd(self, *keys:Any)->'OrderedDictLogger':
        if not self._yaml_log:
            return self

        """Creates new path as specified by the sequence of the keys"""
        self._paths.append([str(k) for k in keys])
        self._stack.append(None) # delay create

        return self # this allows calling __enter__

    def popd(self):
        if not self._yaml_log:
            return

        if len(self._stack)==1:
            raise RuntimeError('There is no child logger, popd() call is invalid')
        self._stack.pop()
        self._paths.pop()

    def path(self)->str:
        if not self._yaml_log:
            return '/'

        # flatten array of array
        return '/'.join(itertools.chain.from_iterable(self._paths[1:]))

    def __enter__(self)->'OrderedDictLogger':
        return self
    def __exit__(self, type, value, traceback):
        self.popd()

    def __contains__(self, key:Any):
        return key in self._cur()

    def __len__(self)->int:
        return len(self._cur())
