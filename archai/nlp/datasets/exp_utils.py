# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import datetime
import logging
import os
import shutil
import signal
import sys
import time
from typing import Optional, Tuple

import nvdllogger
import torch.utils.collect_env

from archai.nlp.datasets.distributed_utils import distributed as nv_distributed

from archai.common import utils, common

try:
    from apex import amp
except ModuleNotFoundError:
    logging.warn('APEX AMP is unavailable')


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self, warmup=0, keep=False):
        self.reset()
        self.warmup = warmup
        self.keep = keep

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.iters = 0
        self.vals = []

    def update(self, val, n=1):
        self.iters += 1
        self.val = val

        if self.iters > self.warmup:
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
            if self.keep:
                self.vals.append(val)


class TimeoutHandler:
    def __init__(self, sig=signal.SIGTERM):
        self.sig = sig

    def __enter__(self):
        self.interrupted = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)

        def handler(signum, frame):
            self.release()
            self.interrupted = True
            logging.info(f'Received SIGTERM')

        signal.signal(self.sig, handler)
        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True


def register_ignoring_timeout_handler(sig=signal.SIGTERM):
    def handler(signum, frame):
        logging.info('Received SIGTERM, ignoring')
    signal.signal(sig, handler)


def log_env_info():
    """
    Prints information about execution environment.
    """
    logging.info('Collecting environment information...')
    env_info = torch.utils.collect_env.get_pretty_env_info()
    logging.info(f'{env_info}')


def benchmark(test_perplexity=None, target_perplexity=None,
              test_throughput=None, target_throughput=None):
    def test(achieved, target, name, higher_better=True):
        passed = True
        if target is not None and achieved is not None:
            logging.info(f'{name} achieved: {achieved:.2f} '
                         f'target: {target:.2f}')
            if higher_better:
                result = (achieved >= target)
            else:
                result = (achieved <= target)

            if result:
                logging.info(f'{name} test passed')
            else:
                logging.info(f'{name} test failed')
                passed = False
        return passed

    passed = True
    passed &= test(test_perplexity, target_perplexity, 'Perplexity', False)
    passed &= test(test_throughput, target_throughput, 'Throughput')
    return passed


def setup_logging(log_all_ranks=True, filename=os.devnull, filemode='w'):
    """
    Configures logging.
    By default logs from all workers are printed to the console, entries are
    prefixed with "N: " where N is the rank of the worker. Logs printed to the
    console don't include timestaps.
    Full logs with timestamps are saved to the log_file file.
    """
    class RankFilter(logging.Filter):
        def __init__(self, rank, log_all_ranks):
            self.rank = rank
            self.log_all_ranks = log_all_ranks

        def filter(self, record):
            record.rank = self.rank
            if self.log_all_ranks:
                return True
            else:
                return (self.rank == 0)

    rank = nv_distributed.get_rank()
    rank_filter = RankFilter(rank, log_all_ranks)

    if log_all_ranks:
        logging_format = "%(asctime)s - %(levelname)s - %(rank)s - %(message)s"
    else:
        logging_format = "%(asctime)s - %(levelname)s - %(message)s"
        if rank != 0:
            filename = os.devnull

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()

    logging.basicConfig(level=logging.DEBUG,
                        format=logging_format,
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=filename,
                        filemode=filemode)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    if log_all_ranks:
        formatter = logging.Formatter('%(rank)s: %(message)s')
    else:
        formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('').addFilter(rank_filter)


def setup_dllogger(enabled=True, filename=os.devnull, disable_multiple=False):
    rank = nv_distributed.get_rank()

    if disable_multiple:
        return
        
    if enabled and rank == 0:
        backends = [
            nvdllogger.JSONStreamBackend(
                nvdllogger.Verbosity.VERBOSE,
                filename,
                ),
            ]
        nvdllogger.init(backends)
    else:
        nvdllogger.init([])


def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
    os.makedirs(dir_path, exist_ok=True)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        os.makedirs(script_path, exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def build_work_dir_name(work_dir, dataset, append_dataset, append_time):
    if append_dataset:
        work_dir = '{}-{}'.format(work_dir, dataset)

    if append_time:
        now = int(time.time())
        now_max = nv_distributed.all_reduce_item(now, op='max')
        now_str = datetime.datetime.fromtimestamp(now_max).strftime('%Y%m%d-%H%M%S')

        work_dir = os.path.join(work_dir, now_str)
    return work_dir


def l2_promote():
    if not utils.is_windows():
        _libcudart = ctypes.CDLL('libcudart.so')
        # Set device limit on the current device
        # cudaLimitMaxL2FetchGranularity = 0x05
        pValue = ctypes.cast((ctypes.c_int*1)(), ctypes.POINTER(ctypes.c_int))
        _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
        _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
        assert pValue.contents.value == 128

def dataset_dir_name(dataset:str)->str:
    if dataset=='wt103':
        return 'wikitext-103'
    if dataset=='wt2':
        return 'wikitext-2'
    if dataset.startswith('olx_'):
        return dataset
    if dataset=='lm1b':
        return 'one-billion-words'
    if dataset=='enwik8':
        raise RuntimeError(f'dataset "{dataset}" is not supported yet')
    if dataset=='text8':
        raise RuntimeError(f'dataset "{dataset}" is not supported yet')
    raise RuntimeError(f'dataset "{dataset}" is not known')

def get_create_dirs(dataroot:Optional[str], dataset_name:str,
                    experiment_name='nv_xformer_xl', output_dir='~/logdir',
                    pretrained_path:Optional[str]=None, cache_dir:Optional[str]=None)->Tuple[str,str,str,str,str]:

    pt_data_dir, pt_output_dir = common.pt_dirs()
    if pt_output_dir:
        pt_output_dir = os.path.join(pt_output_dir, experiment_name)
    dataroot = dataroot or pt_data_dir or common.default_dataroot()
    dataroot = utils.full_path(dataroot)

    dataset_dir = utils.full_path(os.path.join(dataroot,'textpred', dataset_dir_name(dataset_name)))
    output_dir = utils.full_path(pt_output_dir or \
                        os.path.join(output_dir, experiment_name)
                    , create=True)

    if not os.path.isabs(cache_dir):
        cache_dir = os.path.join(dataset_dir, cache_dir)
    
    cache_dir = utils.full_path(cache_dir, create=True)

    if not os.path.isabs(pretrained_path) and pretrained_path:
        pretrained_path = os.path.join(os.path.dirname(output_dir), pretrained_path)

    return dataset_dir, output_dir, pretrained_path, cache_dir, dataroot

def script_init():
        # Disable profiling executor
    try:
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
    except AttributeError:
        pass

    # Before we do anything with models, we want to ensure that we get fp16
    # execution of torch.einsum in APEX AMP.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations.
    # Note that running `--apex_amp_opt_level O2` will remove the need for this
    # code, but it is still valid.
    if 'apex' in sys.modules:
        amp.register_half_function(torch, 'einsum')
