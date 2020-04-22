from typing import Optional, Sequence, Tuple, List
import os
import argparse

import torch
from torch.optim.optimizer import Optimizer
from torch import Tensor, nn
from torch.backends import cudnn
import torch.distributed as dist

from archai.common.config import Config

from archai.common import ml_utils, utils
from archai.common.ordereddict_logger import OrderedDictLogger

class ApexUtils:
    def __init__(self, distdir:Optional[str], apex_config:Config)->None:
        logger = self._create_init_logger(distdir)

        # region conf vars
        self._enabled = apex_config['enabled'] # global switch to disable anything apex
        self._opt_level = apex_config['opt_level'] # optimization level for mixed precision
        self._bn_fp32 = apex_config['bn_fp32'] # keep BN in fp32
        self._loss_scale = apex_config['loss_scale'] # loss scaling mode for mixed prec
        self._sync_bn = apex_config['sync_bn'] # should be replace BNs with sync BNs for distributed model
        self._distributed = apex_config['distributed'] # enable/disable distributed mode
        self._scale_lr = apex_config['scale_lr'] # enable/disable distributed mode
        self._min_world_size = apex_config['min_world_size'] # allows to confirm we are indeed in distributed setting
        conf_gpu_ids = apex_config['gpus']
        # endregion

        self.gpu_ids = [int(i) for i in conf_gpu_ids.split(',') if i]
        self._amp, self._ddp = None, None
        self._gpu = self.gpu_ids[0] if len(self.gpu_ids) else 0 # which GPU to use, we will use only 1 GPU
        self._world_size = 1 # total number of processes in distributed run
        self.local_rank = 0
        self.global_rank = 0

        logger.info({'apex_config': apex_config.to_dict()})
        logger.info({'torch.distributed_is_available': dist.is_available()})
        if dist.is_available():
            logger.info({'gloo_available': dist.is_gloo_available(),
                        'mpi_available': dist.is_mpi_available(),
                        'nccl_available': dist.is_nccl_available()})

        if self._enabled:
            # init enable mixed precision
            assert cudnn.enabled, "Amp requires cudnn backend to be enabled."
            from apex import amp
            self._amp = amp

            # enable distributed processing
            if self._distributed:
                from apex import parallel
                self._ddp = parallel

                assert dist.is_available() # distributed module is available
                assert dist.is_nccl_available()
                dist.init_process_group(backend='nccl', init_method='env://')
                assert dist.is_initialized()

                self._op_map = {'mean': dist.ReduceOp.SUM, 'sum': dist.ReduceOp.SUM,
                            'min': dist.ReduceOp.MIN, 'max': dist.ReduceOp.MAX}

                self._set_ranks()

        assert self._world_size >= 1
        assert not self._min_world_size or self._world_size >= self._min_world_size
        assert self.local_rank >= 0 and self.local_rank < self._world_size
        assert self.global_rank >= 0 and self.global_rank < self._world_size

        assert self._gpu < torch.cuda.device_count()
        torch.cuda.set_device(self._gpu)
        self.device = torch.device('cuda', self._gpu)

        logger.info({'amp_available': self._amp is not None, 'distributed_available': self._distributed is not None})
        logger.info({'distributed': self._distributed, 'world_size': self._world_size,
                     'gpu': self._gpu, 'gpu_ids':self.gpu_ids, 'local_rank': self.local_rank})

        logger.info({'dist_initialized': dist.is_initialized() if dist.is_available() else False})

        logger.close()


    def _set_ranks(self)->None:
        if 'WORLD_SIZE' in os.environ:
            self._world_size = int(os.environ['WORLD_SIZE'])
            assert dist.get_world_size() == self._world_size
        else:
            raise RuntimeError('WORLD_SIZE must be set by distributed launcher when distributed mode is enabled')

        if 'LOCAL_RANK' in os.environ:
            self.local_rank = int(os.environ['LOCAL_RANK'])
        else:
            raise RuntimeError('LOCAL_RANK must be set by distributed launcher when distributed mode is enabled')

        self.global_rank = dist.get_rank()
        # parser = argparse.ArgumentParser()
        # parser.add_argument('--local-rank', type=int, help='local-rank must be supplied by torch distributed launcher!')
        # args, extra_args = parser.parse_known_args()
        # self.local_rank = args.local_rank

        assert self.local_rank < torch.cuda.device_count()
        self._gpu = self.local_rank % torch.cuda.device_count()
        # remap if GPU IDs are specified
        if len(self.gpu_ids):
            assert len(self.gpu_ids) > self.local_rank
            self._gpu = self.gpu_ids[self.local_rank]


    def _create_init_logger(self, distdir:Optional[str])->OrderedDictLogger:
        # create PID specific logger to support many distributed processes
        init_log_filepath, yaml_log_filepath = None, None
        if distdir:
            init_log_filepath = os.path.join(utils.full_path(distdir),
                                            'apex_' + str(os.getpid()) + '.log')
            yaml_log_filepath = os.path.join(utils.full_path(distdir),
                                            'apex_' + str(os.getpid()) + '.yaml')

        sys_logger = utils.create_logger(filepath=init_log_filepath)
        if not init_log_filepath:
            sys_logger.warn('logdir not specified, no logs will be created or any models saved')

        logger = OrderedDictLogger(yaml_log_filepath, sys_logger)

        return logger


    def set_replica_logger(self, logger:OrderedDictLogger)->None:
        # To avoid circular dependency we don't reference logger in common.
        # Furthermore, each replica has its own logger but sharing same exp directory.
        # We can't create replica specific logger at time of init so this is set later.
        self.logger = logger

    def is_mixed(self)->bool:
        return self._amp is not None
    def is_dist(self)->bool:
        return self._ddp is not None
    def is_master(self)->bool:
        return self.global_rank == 0

    def sync_devices(self)->None:
        if self._distributed:
            torch.cuda.synchronize()

    def reduce(self, val, op='mean'):
        if self._distributed:
            if not isinstance(val, Tensor):
                rt = torch.tensor(val)
            else:
                rt = val.clone()

            r_op = self._op_map[op]
            torch.dist.all_reduce(rt, op=r_op)
            if op=='mean':
                rt /= self._world_size
            return rt
        else:
            return val

    def backward(self, loss:torch.Tensor, optim:Optimizer)->None:
        if self._amp:
            with self._amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def to_amp(self, model:nn.Module, optim:Optimizer, batch_size:int)\
                ->Tuple[nn.Module, Optimizer]:
        # conver BNs to sync BNs in distributed mode
        if self._ddp and self._sync_bn:
            model = self._ddp.convert_syncbn_model(model)
            self.logger.info({'BNs_converted': True})

        model = model.to(self.device)

        if self._amp:
            # scale LR
            if self._scale_lr:
                lr = ml_utils.get_optim_lr(optim)
                scaled_lr = lr * self._world_size / float(batch_size)
                ml_utils.set_optim_lr(optim, scaled_lr)
                self.logger.info({'lr_scaled': True, 'old_lr': lr, 'new_lr': scaled_lr})

            model, optim = self._amp.initialize(
                model, optim, opt_level=self._opt_level,
                keep_batchnorm_fp32=self._bn_fp32, loss_scale=self._loss_scale
            )

        if self._ddp:
            # By default, apex.parallel.DistributedDataParallel overlaps communication with
            # computation in the backward pass.
            # delay_allreduce delays all communication to the end of the backward pass.
            model = self._ddp.DistributedDataParallel(model, delay_allreduce=True)

        return model, optim

    def clip_grad(self, clip:float, model:nn.Module, optim:Optimizer)->None:
        if clip > 0.0:
            if self._amp:
                nn.utils.clip_grad_norm_(self._amp.master_params(optim), clip)
            else:
                nn.utils.clip_grad_norm_(model.parameters(), clip)

    def state_dict(self):
        if self._amp:
            return self._amp.state_dict()
        else:
            return None

    def load_state_dict(self, state_dict):
        if self._amp:
            self._amp.load_state_dict()
        else:
            if state_dict is not None:
                raise RuntimeError('checkpoint state_dict is not None but Nvidia apex (amp) is not ')
            else:
                pass

