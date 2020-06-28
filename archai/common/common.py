# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import numpy as np
import os
from typing import List, Iterable, Union, Optional, Tuple
import atexit
import subprocess
import datetime
import yaml
import sys

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import psutil

from .config import Config
from . import utils
from .ordereddict_logger import OrderedDictLogger
from .apex_utils import ApexUtils

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
    return Config.get_inst()

def get_conf_common()->Config:
    return get_conf()['common']

def get_conf_dataset()->Config:
    return get_conf()['dataset']

def get_experiment_name()->str:
    return get_conf_common()['experiment_name']

def get_expdir()->Optional[str]:
    return get_conf_common()['expdir']

def get_datadir()->Optional[str]:
    return get_conf()['dataset']['dataroot']

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
                param_args: list = [], log_level=logging.INFO, use_args=True)->Config:

    # get cloud dirs if any
    pt_data_dir, pt_output_dir, param_overrides = _setup_pt(param_args)

    # init config
    conf = Config(config_filepath=config_filepath,
                  param_args=param_overrides,
                  use_args=use_args)
    Config.set_inst(conf)

    # create experiment dir
    _setup_dirs()

    # validate and log dirs
    expdir = get_expdir()
    assert not pt_output_dir or not expdir.startswith(utils.full_path('~/logdir'))
    logger.info({'expdir': expdir,
                 'PT_DATA_DIR': pt_data_dir, 'PT_OUTPUT_DIR': pt_output_dir})

    # create a[ex to know distributed processing paramters
    conf_apex = get_conf_common()['apex']
    apex = ApexUtils(conf_apex, None)

    # create global logger
    _setup_logger(apex)
    # create info file for current system
    _create_sysinfo(conf)

    # setup tensorboard
    global _tb_writer
    _tb_writer = _create_tb_writer(apex.is_master())

    # create hooks to execute code when script exits
    global _atexit_reg
    if not _atexit_reg:
        atexit.register(on_app_exit)
        _atexit_reg = True

    return conf

def _create_sysinfo(conf:Config)->None:
    expdir = get_expdir()

    if expdir and not utils.is_debugging():
        # copy net config to experiment folder for reference
        with open(expdir_abspath('config_used.yaml'), 'w') as f:
            yaml.dump(conf.to_dict(), f)
        if not utils.is_debugging():
            sysinfo_filepath = expdir_abspath('sysinfo.txt')
            subprocess.Popen([f'./sysinfo.sh "{expdir}" > "{sysinfo_filepath}"'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            shell=True)

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

        # directory for non-master replica logs
        distdir = os.path.join(expdir, 'dist')
        os.makedirs(distdir, exist_ok=True)
    else:
        raise RuntimeError('The logdir setting must be specified for the output directory in yaml')

    # update conf so everyone gets expanded full paths from here on
    # set environment variable so it can be referenced in paths used in config
    os.environ['logdir'] = conf_common['logdir'] = logdir
    os.environ['dataroot'] = conf_data['dataroot'] = dataroot
    os.environ['expdir'] = conf_common['expdir'] = expdir
    os.environ['distdir'] = conf_common['distdir'] = distdir


def _setup_logger(apex:ApexUtils):
    global logger
    logger.close()  # close any previous instances

    conf_common = get_conf_common()
    expdir = conf_common['expdir']
    distdir = conf_common['distdir']

    global_rank = apex.global_rank

    # file where logger would log messages
    if apex.is_master():
        sys_log_filepath = utils.full_path(os.path.join(expdir, 'logs.log'))
        logs_yaml_filepath = utils.full_path(os.path.join(expdir, 'logs.yaml'))
        experiment_name = get_experiment_name()
        enable_stdout = True
    else:
        sys_log_filepath = utils.full_path(os.path.join(distdir, f'logs_{global_rank}.log'))
        logs_yaml_filepath = utils.full_path(os.path.join(distdir, f'logs_{global_rank}.yaml'))
        experiment_name = get_experiment_name() + '_' + str(global_rank)
        enable_stdout = False
    print(f'log_global_rank={global_rank}, log_stdout={sys_log_filepath}, log_file={sys_log_filepath}')

    sys_logger = utils.create_logger(filepath=sys_log_filepath,
                                     name=experiment_name,
                                     enable_stdout=enable_stdout)
    if not sys_log_filepath:
        sys_logger.warn(
            'logdir not specified, no logs will be created or any models saved')

    # reset to new file path
    logger.reset(logs_yaml_filepath, sys_logger)
    logger.info({'command_line': ' '.join(sys.argv[1:])})
    logger.info({
        'datetime:': datetime.datetime.now(),
        'logger_global_rank': global_rank,
        'logger_enable_stdout': enable_stdout,
        'sys_log_filepath': sys_log_filepath
    })





