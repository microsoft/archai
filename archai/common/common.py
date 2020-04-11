import logging
import numpy as np
import os
from typing import List, Iterable, Union, Optional, Tuple
import atexit
import subprocess

import yaml

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

from .config import Config
from .stopwatch import StopWatch
from . import utils
from .ordereddict_logger import OrderedDictLogger

class SummaryWriterDummy:
    def __init__(self, log_dir):
        pass

    def add_scalar(self, *args, **kwargs):
        pass
    def flush(self):
        pass

SummaryWriterAny = Union[SummaryWriterDummy, SummaryWriter]
logger = OrderedDictLogger(None, None)
_tb_writer: SummaryWriterAny = None
_atexit_reg = False # is hook for atexit registered?

def get_conf()->Config:
    return Config.get()

def get_conf_common()->Config:
    return get_conf()['common']

def get_conf_dataset()->Config:
    return get_conf()['dataset']

def get_experiment_name()->str:
    return get_conf_common()['experiment_name']

def get_expdir()->Optional[str]:
    return get_conf_common()['expdir']

def get_logger() -> OrderedDictLogger:
    global logger
    if logger is None:
        raise RuntimeError('get_logger call made before logger was setup!')
    return logger

def get_tb_writer() -> SummaryWriterAny:
    global _tb_writer
    return _tb_writer

def on_app_exit():
    writer = get_tb_writer()
    writer.flush()
    if isinstance(logger, OrderedDictLogger):
        logger.close()

def _setup_pt(param_args: list)->Tuple[str,str, list]:
    # support for pt infrastructure
    pt_data_dir = os.environ.get('PT_DATA_DIR', '')

    # currently yaml should be copying dataset folder to local dir
    # so below is not needed. The hope is that less reads from cloud
    # storage will reduce overall latency.

    # if pt_data_dir:
    #     param_args = ['--nas.eval.loader.dataset.dataroot', pt_data_dir,
    #                   '--nas.search.loader.dataset.dataroot', pt_data_dir,
    #                   '--nas.search.seed_train.loader.dataset.dataroot', pt_data_dir,
    #                   '--nas.search.post_train.loader.dataset.dataroot', pt_data_dir,
    #                   '--autoaug.loader.dataset.dataroot', pt_data_dir] + param_args

    pt_output_dir = os.environ.get('PT_OUTPUT_DIR', '')
    if pt_output_dir:
        # prepend so if supplied from outside it takes back seat
        param_args = ['--common.logdir', pt_output_dir] + param_args

    return pt_data_dir, pt_output_dir, param_args

# initializes random number gen, debugging etc
def common_init(config_filepath: Optional[str]=None,
                param_args: list = [],
                log_level=logging.INFO, is_master=True, use_args=True) \
        -> Config:

    pt_data_dir, pt_output_dir, param_args = _setup_pt(param_args)

    conf = Config(config_filepath=config_filepath,
                  param_args=param_args,
                  use_args=use_args)
    Config.set(conf)

    sw = StopWatch()
    StopWatch.set(sw)

    expdir = _setup_dirs()
    _setup_logger()

    assert not pt_output_dir or not expdir.startswith(utils.full_path('~/logdir'))
    logger.info({'expdir': expdir,
                 'PT_DATA_DIR': pt_data_dir, 'PT_OUTPUT_DIR': pt_output_dir})

    _setup_gpus()

    if expdir:
        # copy net config to experiment folder for reference
        with open(expdir_abspath('config_used.yaml'), 'w') as f:
            yaml.dump(conf.to_dict(), f)
        if not utils.is_debugging():
            sysinfo_filepath = expdir_abspath('sysinfo.txt')
            subprocess.Popen([f'./sysinfo.sh "{expdir}" > "{sysinfo_filepath}"'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            shell=True)

    global _tb_writer
    _tb_writer = _create_tb_writer(is_master)

    global _atexit_reg
    if not _atexit_reg:
        atexit.register(on_app_exit)
        _atexit_reg = True

    return conf

def expdir_abspath(path:str, create=False)->str:
    """Returns full path for given relative path within experiment directory."""
    return utils.full_path(os.path.join('$expdir',path), create=create)

def _create_tb_writer(is_master=True)-> SummaryWriterAny:
    conf_common = get_conf_common()

    tb_dir, conf_enable_tb = utils.full_path(conf_common['tb_dir']), conf_common['tb_enable']
    tb_enable = conf_enable_tb and is_master and tb_dir is not None and len(tb_dir) > 0

    logger.info({'conf_enable_tb': conf_enable_tb,
                 'tb_enable': tb_enable,
                 'tb_dir': tb_dir})

    WriterClass = SummaryWriter if tb_enable else SummaryWriterDummy

    return WriterClass(log_dir=tb_dir)

def _setup_dirs()->Optional[str]:
    conf_common = get_conf_common()
    conf_data = get_conf_dataset()
    experiment_name = get_experiment_name()

    # make sure dataroot exists
    dataroot = utils.full_path(conf_data['dataroot'])
    os.makedirs(dataroot, exist_ok=True)

    # make sure logdir and expdir exists
    logdir = conf_common['logdir']
    if logdir:
        logdir = utils.full_path(logdir)
        expdir = os.path.join(logdir, experiment_name)
        os.makedirs(expdir, exist_ok=True)
    else:
        raise RuntimeError('The logdir setting must be specified for the output directory in yaml')

    # update conf so everyone gets expanded full paths from here on
    conf_common['logdir'], conf_data['dataroot'], conf_common['expdir'] = \
        logdir, dataroot, expdir

    # set environment variable so it can be referenced in paths used in config
    os.environ['expdir'] = expdir

    return expdir

def _setup_logger():
    global logger
    logger.close()  # close any previous instances

    experiment_name = get_experiment_name()

    # file where logger would log messages
    sys_log_filepath = expdir_abspath('logs.log')
    sys_logger = utils.setup_logging(filepath=sys_log_filepath, name=experiment_name)
    if not sys_log_filepath:
        sys_logger.warn(
            'logdir not specified, no logs will be created or any models saved')

    # reset to new file path
    logs_yaml_filepath = expdir_abspath('logs.yaml')
    logger.reset(logs_yaml_filepath, sys_logger)

def _setup_gpus():
    conf_common = get_conf_common()

    if conf_common['gpus'] is not None:
        csv = str(conf_common['gpus'])
        #os.environ['CUDA_VISIBLE_DEVICES'] = str(conf_common['gpus'])
        torch.cuda.set_device(int(csv.split(',')[0]))
        logger.info({'gpu_ids': conf_common['gpus']})
        # alternative: torch.cuda.set_device(config.gpus[0])

    utils.setup_cuda(conf_common['seed'])

    if conf_common['detect_anomaly']:
        logger.warn({'set_detect_anomaly':True})
        torch.autograd.set_detect_anomaly(True)

    logger.info({'gpu_names': utils.cuda_device_names(),
                 'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES']
                     if 'CUDA_VISIBLE_DEVICES' in os.environ else 'NotSet',
                'cudnn.enabled': cudnn.enabled,
                'cudnn.benchmark': cudnn.benchmark,
                'cudnn.deterministic': cudnn.deterministic
                 })

    # gpu_usage = os.popen(
    #     'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
    # ).read().split('\n')
    # for i, line in enumerate(gpu_usage):
    #     vals = line.split(',')
    #     if len(vals) == 2:
    #         logger.info('GPU {} mem: {}, used: {}'.format(i, vals[0], vals[1]))

