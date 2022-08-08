# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Iterable, Type, MutableMapping, Mapping, Any, Optional, Tuple, List, Union, Sized
import  numpy as np
import logging
import csv
from collections import OrderedDict
import sys
import  os
import functools
import pathlib
from pathlib import Path
import random
from itertools import zip_longest
import shutil
import multiprocessing
from distutils import dir_util
from datetime import datetime
import platform
from urllib.parse import urlparse, unquote
from urllib.request import url2pathname

import  torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.optim import lr_scheduler, SGD, Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _WeightedLoss, _Loss
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision.datasets import utils as tvutils

import yaml
import subprocess

class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0.
        self.sum = 0.
        self.cnt = 0
        self.last = 0.

    def update(self, val, n=1):
        self.last = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def first_or_default(it:Iterable, default=None):
    for i in it:
        return i
    return default

def deep_update(d:MutableMapping, u:Mapping, map_type:Type[MutableMapping]=dict)\
        ->MutableMapping:
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = deep_update(d.get(k, map_type()), v, map_type)
        else:
            d[k] = v
    return d

def state_dict(val)->Mapping:
    assert hasattr(val, '__dict__'), 'val must be object with __dict__ otherwise it cannot be loaded back in load_state_dict'

    # Can't do below because val has state_dict() which calls utils.state_dict
    # if has_method(val, 'state_dict'):
    #     d = val.state_dict()
    #     assert isinstance(d, Mapping)
    #     return d

    return {'yaml': yaml.dump(val)}

def load_state_dict(val:Any, state_dict:Mapping)->None:
    assert hasattr(val, '__dict__'), 'val must be object with __dict__'

    # Can't do below because val has state_dict() which calls utils.state_dict
    # if has_method(val, 'load_state_dict'):
    #     return val.load_state_dict(state_dict)

    s = state_dict.get('yaml', None)
    assert s is not None, 'state_dict must contain yaml key'

    obj = yaml.load(s, Loader=yaml.Loader)
    for k, v in obj.__dict__.items():
        setattr(val, k, v)

def deep_comp(o1:Any, o2:Any)->bool:
    # NOTE: dict don't have __dict__
    o1d = getattr(o1, '__dict__', None)
    o2d = getattr(o2, '__dict__', None)

    # if both are objects
    if o1d is not None and o2d is not None:
        # we will compare their dictionaries
        o1, o2 = o1.__dict__, o2.__dict__

    if o1 is not None and o2 is not None:
        # if both are dictionaries, we will compare each key
        if isinstance(o1, dict) and isinstance(o2, dict):
            for k in set().union(o1.keys(), o2.keys()):
                if k in o1 and k in o2:
                    if not deep_comp(o1[k], o2[k]):
                        return False
                else:
                    return False # some key missing
            return True
    # mismatched object types or both are scalers, or one or both None
    return o1 == o2

# We setup env variable if debugging mode is detected for vs_code_debugging.
# The reason for this is that when Python multiprocessing is used, the new process
# spawned do not inherit 'pydevd' so those process do not get detected as in debugging mode
# even though they are. So we set env var which does get inherited by sub processes.
if 'pydevd' in sys.modules:
    os.environ['vs_code_debugging'] = 'True'
def is_debugging()->bool:
    return 'vs_code_debugging' in os.environ and os.environ['vs_code_debugging']=='True'

def full_path(path:str, create=False)->str:
    assert path
    path = os.path.realpath(
            os.path.expanduser(
                os.path.expandvars(path)))
    if create:
        os.makedirs(path, exist_ok=True)
    return path

def zero_file(filepath)->None:
    """Creates or truncates existing file"""
    open(filepath, 'w').close()

def write_string(filepath:str, content:str)->None:
    pathlib.Path(filepath).write_text(content)
def read_string(filepath:str)->str:
    return pathlib.Path(filepath).read_text(encoding='utf-8')

def create_logger(filepath:Optional[str]=None,
                  name:Optional[str]=None,
                  level=logging.INFO,
                  enable_stdout=True)->logging.Logger:
    logging.basicConfig(level=level) # this sets level for standard logging.info calls
    logger = logging.getLogger(name=name)

    # close current handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    logger.setLevel(level)

    if enable_stdout:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter('%(asctime)s %(message)s', '%H:%M'))
        logger.addHandler(ch)

    logger.propagate = False # otherwise root logger prints things again

    if filepath:
        filepath = full_path(filepath)
        # log files gets appeneded if already exist
        # zero_file(filepath)
        fh = logging.FileHandler(filename=full_path(filepath))
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
        logger.addHandler(fh)
    return logger

def fmt(val:Any)->str:
    if isinstance(val, float):
        return f'{val:.4g}'
    return str(val)

def append_csv_file(filepath:str, new_row:List[Tuple[str, Any]], delimiter='\t'):
    fieldnames, rows = [], []

    # get existing field names and rows
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            dr = csv.DictReader(f, delimiter=delimiter)
            fieldnames = dr.fieldnames
            rows = [row for row in dr.reader]
    if fieldnames is None:
        fieldnames = []

    # add field names from old file and new row
    new_fieldnames = OrderedDict([(fn, None) for fn, v in new_row])
    for fn in fieldnames:
        new_fieldnames[fn]=None

    # write new CSV file
    with open(filepath, 'w', newline='') as f:
        dr = csv.DictWriter(f, fieldnames=new_fieldnames.keys(), delimiter=delimiter)
        dr.writeheader()
        for row in rows:
            d = dict((k,v) for k,v in zip(fieldnames, row))
            dr.writerow(d)
        dr.writerow(OrderedDict(new_row))

def has_method(o, name):
    return callable(getattr(o, name, None))

def extract_tar(src, dest=None, gzip=None, delete=False):
    import tarfile

    if dest is None:
        dest = os.path.dirname(src)
    if gzip is None:
        gzip = src.lower().endswith('.gz')

    mode = 'r:gz' if gzip else 'r'
    with tarfile.open(src, mode) as tarfh:
        tarfh.extractall(path=dest)

    if delete:
        os.remove(src)

def download_and_extract_tar(url, download_root, extract_root=None, filename=None,
                             md5=None, **kwargs):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if filename is None:
        filename = os.path.basename(url)

    if not tvutils.check_integrity(os.path.join(download_root, filename), md5):
        tvutils.download_url(url, download_root, filename=filename, md5=md5)

    extract_tar(os.path.join(download_root, filename), extract_root, **kwargs)

def setup_cuda(seed:Union[float, int], local_rank:int=0):
    seed = int(seed) + local_rank
    # setup cuda
    cudnn.enabled = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    #torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True # set to false if deterministic
    torch.set_printoptions(precision=10)
    #cudnn.deterministic = False
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()

def cuda_device_names()->str:
    return ', '.join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

def exec_shell_command(command:str, print_command_start=True, print_command_end=True)->subprocess.CompletedProcess:
    if print_command_start:
        print(f'[{datetime.now()}] Running: {command}')

    ret = subprocess.run(command, shell=True, check=True)

    if print_command_end:
        print(f'[{datetime.now()}] returncode={ret.returncode} Finished: {command}')

    return ret

def zip_eq(*iterables):
    sentinel = object()
    for count, combo in enumerate(zip_longest(*iterables, fillvalue=sentinel)):
        if any(True for c in combo if sentinel is c):
            shorter_its = ','.join([str(i) for i,c in enumerate(combo) if sentinel is c])
            raise ValueError(f'Iterator {shorter_its} have length {count} which is shorter than others')
        yield combo

def dir_downloads()->str:
    return full_path(str(os.path.join(pathlib.Path.home(), "Downloads")))

def filepath_without_ext(filepath:str)->str:
    """Returns '/a/b/c/d.e' for '/a/b/c/d.e.f' """
    return str(pathlib.Path(filepath).with_suffix(''))

def filepath_ext(filepath:str)->str:
    """Returns 'd.e.f' for '/a/b/c/d.e.f' """
    return pathlib.Path(filepath).suffix

def filepath_name_ext(filepath:str)->str:
    """Returns '.f' for '/a/b/c/d.e.f' """
    return pathlib.Path(filepath).suffix

def filepath_name_only(filepath:str)->str:
    """Returns 'd.e' for '/a/b/c/d.e.f' """
    return pathlib.Path(filepath).stem

def change_filepath_ext(filepath:str, new_ext:str)->str:
    """Returns '/a/b/c/d.e.g' for filepath='/a/b/c/d.e.f', new_ext='.g' """
    return str(pathlib.Path(filepath).with_suffix(new_ext))

def change_filepath_name(filepath:str, new_name:str, new_ext:Optional[str]=None)->str:
    """Returns '/a/b/c/h.f' for filepath='/a/b/c/d.e.f', new_name='h' """
    ext = new_ext or filepath_ext(filepath)
    return str(pathlib.Path(filepath).with_name(new_name).with_suffix(ext))

def append_to_filename(filepath:str, name_suffix:str, new_ext:Optional[str]=None)->str:
    """Returns '/a/b/c/h.f' for filepath='/a/b/c/d.e.f', new_name='h' """
    ext = new_ext or filepath_ext(filepath)
    name = filepath_name_only(filepath)
    return str(pathlib.Path(filepath).with_name(name+name_suffix).with_suffix(ext))

def copy_file(src_file:str, dest_dir_or_file:str, preserve_metadata=False, use_shutil:bool=True)->str:
    if not use_shutil:
        assert not preserve_metadata
        return copy_file_basic(src_file, dest_dir_or_file)

    # note that copy2 might fail on some Azure blobs if filesyste does not support OS level copystats
    # so use preserve_metadata=True only if absolutely needed for maximum compatibility
    try:
        copy_fn = shutil.copy2 if preserve_metadata else shutil.copy
        return copy_fn(src_file, dest_dir_or_file)
    except OSError as ex:
        if preserve_metadata or ex.errno != 38: # OSError: [Errno 38] Function not implemented
            raise
        return copy_file_basic(src_file, dest_dir_or_file)

def copy_file_basic(src_file:str, dest_dir_or_file:str)->str:
    # try basic python functions
    # first if dest is dir, get dest file name
    if os.path.isdir(dest_dir_or_file):
        dest_dir_or_file = os.path.join(dest_dir_or_file, filepath_name_ext(src_file))
    with open(src_file, 'rb') as src, open(dest_dir_or_file, 'wb') as dst:
        dst.write(src.read())
    return dest_dir_or_file

def copy_dir(src_dir:str, dest_dir:str, use_shutil:bool=True)->None:
    if os.path.isdir(src_dir):
        if use_shutil:
            shutil.copytree(src_dir, dest_dir)
        else:
            if not os.path.isdir(dest_dir):
                os.makedirs(dest_dir)
            files = os.listdir(src_dir)
            for f in files:
                copy_dir(os.path.join(src_dir, f),
                        os.path.join(dest_dir, f), use_shutil=use_shutil)
    else:
        copy_file(src_dir, dest_dir, use_shutil=use_shutil)

if 'main_process_pid' not in os.environ:
    os.environ['main_process_pid'] = str(os.getpid())
def is_main_process()->bool:
    """Returns True if this process was started as main process instead of child process during multiprocessing"""
    return multiprocessing.current_process().name == 'MainProcess' and os.environ['main_process_pid'] == str(os.getpid())
def main_process_pid()->int:
    return int(os.environ['main_process_pid'])
def process_name()->str:
    return multiprocessing.current_process().name

def is_windows()->bool:
    return platform.system()=='Windows'

def path2uri(path:str, windows_non_standard:bool=False)->str:
    uri = pathlib.Path(full_path(path)).as_uri()

    # there is lot of buggy regex based code out there which expects Windows file URIs as
    # file://C/... instead of standard file:///C/...
    # When passing file uri to such code, turn on windows_non_standard
    if windows_non_standard and is_windows():
        uri = uri.replace('file:///', 'file://')
    return uri

def uri2path(file_uri:str, windows_non_standard:bool=False)->str:
    # there is lot of buggy regex based code out there which expects Windows file URIs as
    # file://C/... instead of standard file:///C/...
    # When passing file uri to such code, turn on windows_non_standard
    if windows_non_standard and is_windows():
        file_uri = file_uri.replace('file://', 'file:///')

    parsed = urlparse(file_uri)
    host = "{0}{0}{mnt}{0}".format(os.path.sep, mnt=parsed.netloc)
    return os.path.normpath(
        os.path.join(host, url2pathname(unquote(parsed.path)))
    )

def get_ranks(items:list, key=lambda v:v, reverse=False)->List[int]:
    sorted_t = sorted(zip(items, range(len(items))),
                      key=lambda t: key(t[0]),
                      reverse=reverse)
    sorted_map = dict((t[1], i) for i, t in enumerate(sorted_t))
    return [sorted_map[i] for i in range(len(items))]

def dedup_list(l:List)->List:
    return list(OrderedDict.fromkeys(l))

def delete_file(filepath:str)->bool:
    if os.path.isfile(filepath):
        os.remove(filepath)
        return True
    else:
        return False

def save_as_yaml(obj, filepath:str)->None:
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(obj, f, default_flow_style=False)

def create_file_name_identifier(file_name:Path, identifier:str)->Path:
    return file_name.parent.joinpath(file_name.stem + identifier).with_suffix(file_name.suffix)

def map_to_list(variable:Union[int,float,Sized], size:int)->Sized:
    if isinstance(variable, Sized):
        size_diff = size - len(variable)

        if size_diff < 0:
            return variable[:size]
        elif size_diff == 0:
            return variable
        elif size_diff > 0:
            return variable + [variable[0]] * size_diff

    return [variable] * size

def rsetattr(obj:Any, attr:Any, value:Any)->None:
    # Copyright @ https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties/31174427#31174427
    pre_attr, _, post_attr = attr.rpartition('.')

    return setattr(rgetattr(obj, pre_attr) if pre_attr else obj, post_attr, value)

def rgetattr(obj:Any, attr:Any, *args)->Any:
    # Copyright @ https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties/31174427#31174427
    def _getattr(obj:Any, attr:Any)->Any:
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))

def attr_to_dict(obj:Any, recursive:bool=True)->Dict[str, Any]:
    MAX_LIST_LEN = 10
    variables = {}

    var_dict = dict(vars(obj.__class__))
    try:
        var_dict.update(dict(vars(obj)))
    except TypeError:
        pass

    for k, v in var_dict.items():
        if k[0] == '_':
            continue

        if isinstance(v, (int, float, str)):
            variables[k.lower()] = v

        elif isinstance(v, list) and (len(v) == 0 or isinstance(v[0], (int, float, str))):
            variables[k.lower()] = v[:MAX_LIST_LEN]

        elif isinstance(v, set) and (len(v) == 0 or isinstance(next(iter(v)), (int, float, str))):
            variables[k.lower()] = list(v)[:MAX_LIST_LEN]

        elif recursive:
            settings_fn = getattr(v, 'settings', None)

            if callable(settings_fn):
                variables[k.lower()] = settings_fn()

    return variables
