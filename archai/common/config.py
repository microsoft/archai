# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from typing import Callable, List, Type, Optional, Any, Union
from collections import UserDict
from typing import Sequence
from argparse import ArgumentError
from collections.abc import Mapping, MutableMapping
import os
from distutils.util import strtobool
import copy
from os import stat

import yaml

from . import yaml_utils


# global config instance
_config:'Config' = None

# TODO: remove this duplicate code which is also in utils.py without circular deps
def deep_update(d:MutableMapping, u:Mapping, create_map:Callable[[],MutableMapping])\
        ->MutableMapping:
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = deep_update(d.get(k, create_map()), v, create_map)
        else:
            d[k] = v
    return d

class Config(UserDict):
    def __init__(self, config_filepath:Optional[str]=None,
                 app_desc:Optional[str]=None, use_args=False,
                 param_args: Sequence = [], resolve_redirects=True) -> None:
        """Create config from specified files and args

        Config is simply a dictionary of key, value map. The value can itself be
        a dictionary so config can be hierarchical. This class allows to load
        config from yaml. A special key '__include__' can specify another yaml
        relative file path (or list of file paths) which will be loaded first
        and the key-value pairs in the main file
        will override the ones in include file. You can think of included file as
        defaults provider. This allows to create one base config and then several
        environment/experiment specific configs. On the top of that you can use
        param_args to perform final overrides for a given run.

        Keyword Arguments:
            config_filepath {[str]} -- [Yaml file to load config from, could be names of files separated by semicolon which will be loaded in sequence oveeriding previous config] (default: {None})
            app_desc {[str]} -- [app description that will show up in --help] (default: {None})
            use_args {bool} -- [if true then command line parameters will override parameters from config files] (default: {False})
            param_args {Sequence} -- [parameters specified as ['--key1',val1,'--key2',val2,...] which will override parameters from config file.] (default: {[]})
            resolve_redirects -- [if True then _copy commands in yaml are executed]
        """
        super(Config, self).__init__()

        self.args, self.extra_args = None, []

        if use_args:
            # let command line args specify/override config file
            parser = argparse.ArgumentParser(description=app_desc)
            parser.add_argument('--config', type=str, default=None,
                help='config filepath in yaml format, can be list separated by ;')
            self.args, self.extra_args = parser.parse_known_args()
            config_filepath = self.args.config or config_filepath

        if config_filepath:
            for filepath in config_filepath.strip().split(';'):
                self._load_from_file(filepath.strip())

        # Create a copy of ourselves and do the resolution over it.
        # This resolved_conf then can be used to search for overrides that
        # wouldn't have existed before resolution.
        resolved_conf = copy.deepcopy(self)
        if resolve_redirects:
            yaml_utils.resolve_all(resolved_conf)

        # Let's do final overrides from args
        self._update_from_args(param_args, resolved_conf)      # merge from params
        self._update_from_args(self.extra_args, resolved_conf) # merge from command line

        if resolve_redirects:
            yaml_utils.resolve_all(self)

        self.config_filepath = config_filepath

    def _load_from_file(self, filepath:Optional[str])->None:
        if filepath:
            filepath = os.path.expanduser(os.path.expandvars(filepath))
            filepath = os.path.abspath(filepath)
            with open(filepath, 'r') as f:
                config_yaml = yaml.load(f, Loader=yaml.Loader)
            self._process_includes(config_yaml, filepath)
            deep_update(self, config_yaml, lambda: Config(resolve_redirects=False))
            print('config loaded from: ', filepath)

    def _process_includes(self, config_yaml, filepath:str):
        if '__include__' in config_yaml:
            # include could be file name or array of file names to apply in sequence
            includes = config_yaml['__include__']
            if isinstance(includes, str):
                includes = [includes]
            assert isinstance(includes, List), "'__include__' value must be string or list"
            for include in includes:
                include_filepath = os.path.join(os.path.dirname(filepath), include)
                self._load_from_file(include_filepath)

    def _update_from_args(self, args:Sequence, resolved_section:'Config')->None:
        i = 0
        while i < len(args)-1:
            arg = args[i]
            if arg.startswith(("--")):
                path = arg[len("--"):].split('.')
                i += Config._update_section(self, path, args[i+1], resolved_section)
            else: # some other arg
                i += 1

    def to_dict(self)->dict:
        return deep_update({}, self, lambda: dict()) # type: ignore

    @staticmethod
    def _update_section(section:'Config', path:List[str], val:Any, resolved_section:'Config')->int:
        for p in range(len(path)-1):
            sub_path = path[p]
            if sub_path in resolved_section:
                resolved_section = resolved_section[sub_path]
                if not sub_path in section:
                    section[sub_path] = Config(resolve_redirects=False)
                section = section[sub_path]
            else:
                return 1 # path not found, ignore this
        key = path[-1] # final leaf node value

        if key in resolved_section:
            original_val, original_type = None, None
            try:
                original_val = resolved_section[key]
                original_type = type(original_val)
                if original_type == bool: # bool('False') is True :(
                    original_type = lambda x: strtobool(x)==1
                section[key] = original_type(val)
            except Exception as e:
                raise KeyError(
                    f'The yaml key or command line argument "{key}" is likely not named correctly or value is of wrong data type. Error was occured when setting it to value "{val}".'
                    f'Originally it is set to {original_val} which is of type {original_type}.'
                    f'Original exception: {e}')
            return 2 # path was found, increment arg pointer by 2 as we use up val
        else:
            return 1 # path not found, ignore this

    def get_val(self, key, default_val):
        return super().get(key, default_val)

    @staticmethod
    def set_inst(instance:'Config')->None:
        global _config
        _config = instance

    @staticmethod
    def get_inst()->'Config':
        global _config
        return _config

