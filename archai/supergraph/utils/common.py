# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import atexit
import os
import subprocess
from typing import Optional, Tuple, Union

import yaml
from send2trash import send2trash
from torch.utils.tensorboard.writer import SummaryWriter

from archai.common.config import Config
from archai.common.ordered_dict_logger import get_global_logger
from archai.supergraph.utils import utils
from archai.supergraph.utils.apex_utils import ApexUtils

logger = get_global_logger()


class SummaryWriterDummy:
    def __init__(self, log_dir):
        pass

    def add_scalar(self, *args, **kwargs):
        pass
    def flush(self):
        pass

SummaryWriterAny = Union[SummaryWriterDummy, SummaryWriter]

_tb_writer: Optional[SummaryWriterAny] = None
_atexit_reg = False # is hook for atexit registered?


def get_conf(conf:Optional[Config]=None)->Config:
    if conf is not None:
        return conf
    return Config.get_instance()

def get_conf_common(conf:Optional[Config]=None)->Config:
    return get_conf(conf)['common']

def get_conf_dataset(conf:Optional[Config]=None)->Config:
    return get_conf(conf)['dataset']

def get_experiment_name(conf:Optional[Config]=None)->str:
    return get_conf_common(conf)['experiment_name']

def get_expdir(conf:Optional[Config]=None)->Optional[str]:
    return get_conf_common(conf)['expdir']

def get_datadir(conf:Optional[Config]=None)->Optional[str]:
    return get_conf(conf)['dataset']['dataroot']

def get_tb_writer() -> SummaryWriterAny:
    global _tb_writer
    assert _tb_writer
    return _tb_writer

class CommonState:
    def __init__(self) -> None:
        global _logger, _conf, _tb_writer
        self.logger = get_global_logger()
        self.conf = get_conf()
        self.tb_writer = _tb_writer

def on_app_exit():
    print('Process exit:', os.getpid(), flush=True)
    writer = get_tb_writer()
    writer.flush()

def pt_dirs()->Tuple[str, str]:
    # dirs for pt infrastructure are supplied in env vars

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

    return pt_data_dir, pt_output_dir

def _pt_params(param_args: list)->list:
    pt_data_dir, pt_output_dir = pt_dirs()

    if pt_output_dir:
        # prepend so if supplied from outside it takes back seat
        param_args = ['--common.logdir', pt_output_dir] + param_args

    return param_args

def get_state()->CommonState:
    return CommonState()

def init_from(state:CommonState)->None:
    global _tb_writer

    Config.set_instance(state.conf)

    _tb_writer = state.tb_writer


def create_conf(config_filepath: Optional[str]=None,
                param_args: list = [], use_args=True)->Config:

    # modify passed args for pt infrastructure
    # if pt infrastructure doesn't exit then param_overrides == param_args
    param_overrides = _pt_params(param_args)

    # create env vars that might be used in paths in config
    if 'default_dataroot' not in os.environ:
        os.environ['default_dataroot'] = default_dataroot()

    conf = Config(file_path=config_filepath,
                  args=param_overrides,
                  parse_cl_args=use_args)
    _update_conf(conf)

    return conf


# TODO: rename this simply as init
# initializes random number gen, debugging etc
def common_init(config_filepath: Optional[str]=None,
                param_args: list = [], use_args=True,
                clean_expdir=False)->Config:

    # TODO: multiple child processes will create issues with shared state so we need to
    # detect multiple child processes but allow if there is only one child process.
    # if not utils.is_main_process():
    #     raise RuntimeError('common_init should not be called from child process. Please use Common.init_from()')

    # setup global instance
    conf = create_conf(config_filepath, param_args, use_args)
    Config.set_instance(conf)

    # setup env vars which might be used in paths
    update_envvars(conf)

    # create experiment dir
    create_dirs(conf, clean_expdir)

    _create_sysinfo(conf)

    # create apex to know distributed processing paramters
    conf_apex = get_conf_common(conf)['apex']
    apex = ApexUtils(conf_apex)

    # setup tensorboard
    global _tb_writer
    _tb_writer = create_tb_writer(conf, apex.is_master())

    # create hooks to execute code when script exits
    global _atexit_reg
    if not _atexit_reg:
        atexit.register(on_app_exit)
        _atexit_reg = True

    return conf

def _create_sysinfo(conf:Config)->None:
    expdir = get_expdir(conf)

    if expdir and not utils.is_debugging():
        # copy net config to experiment folder for reference
        with open(expdir_abspath('config_used.yaml'), 'w') as f:
            yaml.dump(conf.to_dict(), f)
        if not utils.is_debugging():
            sysinfo_filepath = expdir_abspath('sysinfo.txt')
            subprocess.Popen([f'./scripts/sysinfo.sh "{expdir}" > "{sysinfo_filepath}"'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            shell=True)

def expdir_abspath(path:str, create=False)->str:
    """Returns full path for given relative path within experiment directory."""
    return utils.full_path(os.path.join('$expdir',path), create=create)

def create_tb_writer(conf:Config, is_master=True)-> SummaryWriterAny:
    conf_common = get_conf_common(conf)

    tb_dir, conf_enable_tb = utils.full_path(conf_common['tb_dir']), conf_common['tb_enable']
    tb_enable = conf_enable_tb and is_master and tb_dir is not None and len(tb_dir) > 0

    logger.info({'conf_enable_tb': conf_enable_tb,
                 'tb_enable': tb_enable,
                 'tb_dir': tb_dir})

    WriterClass = SummaryWriter if tb_enable else SummaryWriterDummy

    return WriterClass(log_dir=tb_dir)

def is_pt()->bool:
    """Is this code running in pt infrastrucuture"""
    return os.environ.get('PT_OUTPUT_DIR', '') != ''

def default_dataroot()->str:
    # the home folder on ITP VMs is super slow so use local temp directory instead
    return '/var/tmp/dataroot' if is_pt() else '~/dataroot'

def _update_conf(conf:Config)->None:
    """Updates conf with full paths resolving enviromental vars"""

    conf_common = get_conf_common(conf)
    conf_dataset = get_conf_dataset(conf)
    experiment_name = conf_common['experiment_name']

    # make sure dataroot exists
    dataroot = conf_dataset['dataroot']
    dataroot = utils.full_path(dataroot)

    # make sure logdir and expdir exists
    logdir = conf_common['logdir']
    if logdir:
        logdir = utils.full_path(logdir)
        expdir = os.path.join(logdir, experiment_name)

        # directory for non-master replica logs
        distdir = os.path.join(expdir, 'dist')
    else:
        expdir = distdir = logdir

    # update conf so everyone gets expanded full paths from here on
    # set environment variable so it can be referenced in paths used in config
    conf_common['logdir'] = logdir
    conf_dataset['dataroot'] = dataroot
    conf_common['expdir'] = expdir
    conf_common['distdir'] = distdir

def update_envvars(conf)->None:
    """Get values from config and put it into env vars"""
    conf_common = get_conf_common(conf)
    logdir = conf_common['logdir']
    expdir = conf_common['expdir']
    distdir = conf_common['distdir']

    conf_dataset = get_conf_dataset(conf)
    dataroot = conf_dataset['dataroot']

    # update conf so everyone gets expanded full paths from here on
    # set environment variable so it can be referenced in paths used in config
    os.environ['logdir'] = logdir
    os.environ['dataroot'] = dataroot
    os.environ['expdir'] = expdir
    os.environ['distdir'] = distdir

def clean_ensure_expdir(conf:Optional[Config], clean_dir:bool, ensure_dir:bool)->None:
    expdir = get_expdir(conf)
    assert expdir
    if clean_dir and os.path.exists(expdir):
        send2trash(expdir)
    if ensure_dir:
        os.makedirs(expdir, exist_ok=True)

def create_dirs(conf:Config, clean_expdir:bool)->Optional[str]:
    conf_common = get_conf_common(conf)
    logdir = conf_common['logdir']
    expdir = conf_common['expdir']
    distdir = conf_common['distdir']

    conf_dataset = get_conf_dataset(conf)
    dataroot = utils.full_path(conf_dataset['dataroot'])

    # make sure dataroot exists
    os.makedirs(dataroot, exist_ok=True)

    # make sure logdir and expdir exists
    if logdir:
        clean_ensure_expdir(conf, clean_dir=clean_expdir, ensure_dir=True)
        os.makedirs(distdir, exist_ok=True)
    else:
        raise RuntimeError('The logdir setting must be specified for the output directory in yaml')

    # get cloud dirs if any
    pt_data_dir, pt_output_dir = pt_dirs()

    # validate dirs
    assert not pt_output_dir or not expdir.startswith(utils.full_path('~/logdir'))

    logger.info({'expdir': expdir,
                 # create info file for current system
                 'PT_DATA_DIR': pt_data_dir, 'PT_OUTPUT_DIR': pt_output_dir})
