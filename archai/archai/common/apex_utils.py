# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Sequence, Tuple, List
import os
import argparse

import torch
from torch.optim.optimizer import Optimizer
from torch import Tensor, nn
from torch.backends import cudnn
import torch.distributed as dist

import ray

import psutil

from archai.common.config import Config
from archai.common import ml_utils, utils
from archai.common.ordereddict_logger import OrderedDictLogger
from archai.common.multi_optim import MultiOptim

class ApexUtils:
    def __init__(self, apex_config:Config, logger:Optional[OrderedDictLogger])->None:
        # region conf vars
        self._enabled = apex_config['enabled'] # global switch to disable anything apex
        self._distributed_enabled = apex_config['distributed_enabled'] # enable/disable distributed mode
        self._mixed_prec_enabled = apex_config['mixed_prec_enabled'] # enable/disable distributed mode
        self._opt_level = apex_config['opt_level'] # optimization level for mixed precision
        self._bn_fp32 = apex_config['bn_fp32'] # keep BN in fp32
        self._loss_scale = apex_config['loss_scale'] # loss scaling mode for mixed prec
        self._sync_bn = apex_config['sync_bn'] # should be replace BNs with sync BNs for distributed model
        self._scale_lr = apex_config['scale_lr'] # enable/disable distributed mode
        self._min_world_size = apex_config['min_world_size'] # allows to confirm we are indeed in distributed setting
        seed = apex_config['seed']
        detect_anomaly = apex_config['detect_anomaly']
        conf_gpu_ids = apex_config['gpus']

        conf_ray = apex_config['ray']
        self.ray_enabled = conf_ray['enabled']
        self.ray_local_mode = conf_ray['local_mode']
        # endregion

        # to avoid circular references= with common, logger is passed from outside
        self.logger = logger

        # defaults for non-distributed mode
        self._amp, self._ddp = None, None
        self._set_ranks(conf_gpu_ids)

        #_log_info({'apex_config': apex_config.to_dict()})
        self._log_info({'ray.enabled':  self.is_ray(), 'apex.enabled': self._enabled})
        self._log_info({'torch.distributed.is_available': dist.is_available(),
                        'apex.distributed_enabled': self._distributed_enabled,
                        'apex.mixed_prec_enabled': self._mixed_prec_enabled})

        if dist.is_available():
            # dist.* properties are otherwise not accessible
            self._op_map = {'mean': dist.ReduceOp.SUM, 'sum': dist.ReduceOp.SUM,
                        'min': dist.ReduceOp.MIN, 'max': dist.ReduceOp.MAX}
            self._log_info({'gloo_available': dist.is_gloo_available(),
                        'mpi_available': dist.is_mpi_available(),
                        'nccl_available': dist.is_nccl_available()})

        if self.is_mixed():
            # init enable mixed precision
            assert cudnn.enabled, "Amp requires cudnn backend to be enabled."
            from apex import amp
            self._amp = amp

        # enable distributed processing
        if self.is_dist():
            assert not self.is_ray(), "Ray is not yet enabled for Apex distributed mode"

            from apex import parallel
            self._ddp = parallel

            assert dist.is_available() # distributed module is available
            assert dist.is_nccl_available()
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl', init_method='env://')
                assert dist.is_initialized()
            assert dist.get_world_size() == self.world_size
            assert dist.get_rank() == self.global_rank

        if self.is_ray():
            assert not self.is_dist(), "Ray is not yet enabled for Apex distributed mode"

            import ray

            if not ray.is_initialized():
                ray.init(local_mode=self.ray_local_mode, include_dashboard=False,
                         # for some reason Ray is detecting wrong number of GPUs
                         num_gpus=torch.cuda.device_count(),
                         # number of CPUs as well
                         num_cpus=psutil.cpu_count())
                ray_cpus = ray.nodes()[0]['Resources']['CPU']
                ray_gpus = ray.nodes()[0]['Resources']['GPU']
                self._log_info({'ray_cpus': ray_cpus, 'ray_gpus':ray_gpus})

        assert self.world_size >= 1
        assert not self._min_world_size or self.world_size >= self._min_world_size
        assert self.local_rank >= 0 and self.local_rank < self.world_size
        assert self.global_rank >= 0 and self.global_rank < self.world_size

        assert self._gpu < torch.cuda.device_count()
        torch.cuda.set_device(self._gpu)
        self.device = torch.device('cuda', self._gpu)
        self._setup_gpus(seed, detect_anomaly)

        self._log_info({'amp_available': self._amp is not None,
                     'distributed_available': self._ddp is not None})
        self._log_info({'dist_initialized': dist.is_initialized() if dist.is_available() else False,
                     'world_size': self.world_size,
                     'gpu': self._gpu, 'gpu_ids':self.gpu_ids,
                     'local_rank': self.local_rank,
                     'global_rank': self.global_rank})


    def _setup_gpus(self, seed:float, detect_anomaly:bool):
        utils.setup_cuda(seed, local_rank=self.local_rank)

        torch.autograd.set_detect_anomaly(detect_anomaly)
        self._log_info({'set_detect_anomaly': detect_anomaly,
                          'is_anomaly_enabled': torch.is_anomaly_enabled()})

        self._log_info({'gpu_names': utils.cuda_device_names(),
                    'gpu_count': torch.cuda.device_count(),
                    'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES']
                        if 'CUDA_VISIBLE_DEVICES' in os.environ else 'NotSet',
                    'cudnn.enabled': cudnn.enabled,
                    'cudnn.benchmark': cudnn.benchmark,
                    'cudnn.deterministic': cudnn.deterministic,
                    'cudnn.version': cudnn.version()
                    })
        self._log_info({'memory': str(psutil.virtual_memory())})
        self._log_info({'CPUs': str(psutil.cpu_count())})

        # gpu_usage = os.popen(
        #     'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
        # ).read().split('\n')
        # for i, line in enumerate(gpu_usage):
        #     vals = line.split(',')
        #     if len(vals) == 2:
        #         _log_info('GPU {} mem: {}, used: {}'.format(i, vals[0], vals[1]))

    def _set_ranks(self, conf_gpu_ids:str)->None:

        # this function needs to work even when torch.distributed is not available

        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])
        else:
            self.world_size = 1

        if 'LOCAL_RANK' in os.environ:
            self.local_rank = int(os.environ['LOCAL_RANK'])
        else:
            self.local_rank = 0

        if 'RANK' in os.environ:
            self.global_rank = int(os.environ['RANK'])
        else:
            self.global_rank = 0

        assert self.local_rank < torch.cuda.device_count(), \
            f'local_rank={self.local_rank} but device_count={torch.cuda.device_count()}' \
            ' Possible cause may be Pytorch is not GPU enabled or you have too few GPUs'

        self.gpu_ids = [int(i) for i in conf_gpu_ids.split(',') if i]
        # which GPU to use, we will use only 1 GPU per process to avoid complications with apex
        # remap if GPU IDs are specified
        if len(self.gpu_ids):
            assert len(self.gpu_ids) > self.local_rank
            self._gpu = self.gpu_ids[self.local_rank]
        else:
            self._gpu = self.local_rank % torch.cuda.device_count()


    def is_mixed(self)->bool:
        return self._enabled and self._mixed_prec_enabled
    def is_dist(self)->bool:
        return self._enabled and self._distributed_enabled
    def is_master(self)->bool:
        return self.global_rank == 0
    def is_ray(self)->bool:
        return self.ray_enabled

    def _log_info(self, d:dict)->None:
        if self.logger is not None:
            self.logger.info(d)

    def sync_devices(self)->None:
        if self.is_dist():
            torch.cuda.synchronize(self.device)
    def barrier(self)->None:
        if self.is_dist():
            dist.barrier() # wait for all processes to come to this point

    def reduce(self, val, op='mean'):
        if self.is_dist():
            if not isinstance(val, Tensor):
                rt = torch.tensor(val).to(self.device)
                converted = True
            else:
                rt = val.clone().to(self.device)
                converted = False

            r_op = self._op_map[op]
            dist.all_reduce(rt, op=r_op)
            if op=='mean':
                rt /= self.world_size

            if converted and len(rt.shape)==0:
                return rt.item()
            return rt
        else:
            return val

    def _get_optim(self, multi_optim:MultiOptim)->Optimizer:
        assert len(multi_optim)==1, \
            'Mixed precision is only supported for one optimizer'  \
            f' but {len(multi_optim)} optimizers were supplied'
        return multi_optim[0].optim

    def backward(self, loss:torch.Tensor, multi_optim:MultiOptim)->None:
        if self.is_mixed():
            optim = self._get_optim(multi_optim)
            with self._amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def to_amp(self, model:nn.Module, multi_optim:MultiOptim, batch_size:int)\
                ->nn.Module:
        # conver BNs to sync BNs in distributed mode
        if self.is_dist() and self._sync_bn:
            model = self._ddp.convert_syncbn_model(model)
            self._log_info({'BNs_converted': True})

        model = model.to(self.device)

        if self.is_mixed():
            optim = self._get_optim(multi_optim)

            # scale LR
            if self.is_dist() and self._scale_lr:
                lr = ml_utils.get_optim_lr(optim)
                scaled_lr = lr * self.world_size / float(batch_size)
                ml_utils.set_optim_lr(optim, scaled_lr)
                self._log_info({'lr_scaled': True, 'old_lr': lr, 'new_lr': scaled_lr})

            model, optim = self._amp.initialize(
                model, optim, opt_level=self._opt_level,
                keep_batchnorm_fp32=self._bn_fp32, loss_scale=self._loss_scale
            )

            # put back amp'd optim
            multi_optim[0].optim = optim

        if self.is_dist():
            # By default, apex.parallel.DistributedDataParallel overlaps communication with
            # computation in the backward pass.
            # delay_allreduce delays all communication to the end of the backward pass.
            model = self._ddp.DistributedDataParallel(model, delay_allreduce=True)

        return model

    def clip_grad(self, clip:float, model:nn.Module, multi_optim:MultiOptim)->None:
        if clip > 0.0:
            if self.is_mixed():
                optim = self._get_optim(multi_optim)
                nn.utils.clip_grad_norm_(self._amp.master_params(optim), clip)
            else:
                nn.utils.clip_grad_norm_(model.parameters(), clip)

    def state_dict(self):
        if self.is_mixed():
            return self._amp.state_dict()
        else:
            return None

    def load_state_dict(self, state_dict):
        if self.is_mixed():
            self._amp.load_state_dict(state_dict)


