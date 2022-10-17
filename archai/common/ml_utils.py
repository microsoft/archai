# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Iterable, Type, MutableMapping, Mapping, Any, Optional, Tuple, List, Sequence
from collections import defaultdict, Counter, OrderedDict
import  numpy as np
import math
import gc

import  torch
from torch import Tensor, nn
from torch.optim import lr_scheduler, SGD, Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _WeightedLoss, _Loss
import torch.nn.functional as F

import statopt

from .config import Config
from .cocob import CocobBackprop
from .ml_losses import SmoothCrossEntropyLoss
from .warmup_scheduler import GradualWarmupScheduler


def create_optimizer(conf_opt:Config, params)->Optimizer:
    optim_type = conf_opt['type']
    lr = conf_opt.get_val('lr', math.nan)
    decay = conf_opt.get_val('decay', math.nan)
    decay_bn = conf_opt.get_val('decay_bn', math.nan) # some optim may not support weight decay

    if not math.isnan(decay_bn):
        bn_params = [v for n, v in params if 'bn' in n]
        rest_params = [v for n, v in params if not 'bn' in n]
        params = [{
            'params': bn_params,
            'weight_decay': decay_bn
        }, {
            'params': rest_params,
            'weight_decay': decay
        }]

    if optim_type == 'sgd':
        return SGD(
           params,
            lr=lr,
            momentum=conf_opt['momentum'],
            weight_decay=decay,
            nesterov=conf_opt['nesterov']
        )
    elif optim_type == 'adam':
         return Adam(params,
            lr=lr,
            betas=conf_opt['betas'],
            weight_decay=decay)
    elif optim_type == 'cocob':
        return CocobBackprop(params,
            alpha=conf_opt['alpha'])
    elif optim_type == 'salsa':
        return statopt.SALSA(params,
            alpha=conf_opt['alpha'])
    else:
        raise ValueError('invalid optimizer type=%s' % optim_type)

def get_optim_lr(optimizer:Optimizer)->float:
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_optim_lr(optimizer:Optimizer, lr:float)->None:
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def ensure_pytorch_ver(min_ver:str, error_msg:str)->bool:
    tv = torch.__version__.split('.')
    req = min_ver.split('.')
    for i,j in zip(tv, req):
        i,j = int(i), int(j)
        if i > j:
            return True
        if i < j:
            if error_msg:
                raise RuntimeError(f'Minimum required PyTorch version is {min_ver} but installed version is {torch.__version__}: {error_msg}')
            return False
    return True

def join_chunks(chunks:List[Tensor])->Tensor:
    """If batch was divided in chunks, this functions joins them again"""
    assert len(chunks)
    if len(chunks) == 1:
        return chunks[0] # nothing to concate
    if len(chunks[0].shape):
        return torch.cat(chunks)
    return torch.stack(chunks) # TODO: this adds new dimension

def create_lr_scheduler(conf_lrs:Config, epochs:int, optimizer:Optimizer,
        steps_per_epoch:Optional[int])-> Tuple[Optional[_LRScheduler], bool]:

    # epoch_or_step - apply every epoch or every step
    scheduler, epoch_or_step = None, True # by default sched step on epoch

    conf_warmup = conf_lrs.get_val('warmup', None)
    warmup_epochs = 0
    if conf_warmup is not None and 'epochs' in conf_warmup:
        warmup_epochs = conf_warmup['epochs']

    if conf_lrs is not None:
        lr_scheduler_type = conf_lrs['type'] # TODO: default should be none?

        if lr_scheduler_type == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                T_max=epochs-warmup_epochs,
                eta_min=conf_lrs['min_lr'])
        elif lr_scheduler_type == 'multi_step':
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=conf_lrs['milestones'],
                                                 gamma=conf_lrs['gamma'])
        elif lr_scheduler_type == 'pyramid':
            scheduler = _adjust_learning_rate_pyramid(optimizer,
                epochs-warmup_epochs,
                get_optim_lr(optimizer))
        elif lr_scheduler_type == 'step':
            decay_period = conf_lrs['decay_period']
            gamma = conf_lrs['gamma']
            scheduler = lr_scheduler.StepLR(optimizer, decay_period, gamma=gamma)
        elif lr_scheduler_type == 'one_cycle':
            assert steps_per_epoch is not None
            ensure_pytorch_ver('1.3.0', 'LR scheduler OneCycleLR is not available.')
            max_lr = conf_lrs['max_lr']
            epoch_or_step = False
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,
                            epochs=epochs-warmup_epochs,
                            steps_per_epoch=steps_per_epoch,
                        )  # TODO: other params
        elif not lr_scheduler_type:
                scheduler = None
        else:
            raise ValueError('invalid lr_schduler=%s' % lr_scheduler_type)

        # select warmup for LR schedule
        if warmup_epochs:
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=conf_lrs['warmup'].get_val('multiplier', 1.0),
                total_epoch=warmup_epochs,
                after_scheduler=scheduler
            )

    return scheduler, epoch_or_step


def _adjust_learning_rate_pyramid(optimizer, max_epoch:int, base_lr:float):
    def _internal_adjust_learning_rate_pyramid(epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = base_lr * (0.1 ** (epoch // (max_epoch * 0.5))) * (0.1 ** (epoch // (max_epoch * 0.75)))
        return lr

    return lr_scheduler.LambdaLR(optimizer, _internal_adjust_learning_rate_pyramid)


def get_lossfn(conf_lossfn:Config)->_Loss:
    type = conf_lossfn['type']
    if type == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif type == 'CrossEntropyLabelSmooth':
        return SmoothCrossEntropyLoss(smoothing=conf_lossfn['smoothing'])
    else:
        raise ValueError('criterian type "{}" is not supported'.format(type))

def param_size(module:nn.Module, ignore_aux=True, only_req_grad=False):
    """count all parameters excluding auxiliary"""
    return np.sum(v.numel() for name, v in module.named_parameters() \
        if (not ignore_aux or ("auxiliary" not in name)) \
            and (not only_req_grad or v.requires_grad))

def save_model(model, model_path):
    #logger.info('saved to model: {}'.format(model_path))
    torch.save(model.state_dict(), model_path)

def load_model(model, model_path):
    #logger.info('load from model: {}'.format(model_path))
    model.load_state_dict(torch.load(model_path))

def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # Bernoulli returns 1 with pobability p and 0 with 1-p.
        # Below generates tensor of shape (batch,1,1,1) filled with 1s and 0s
        #   as per keep_prob.
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob) \
            .to(device=x.device)
        # scale tensor by 1/p as we will be losing other values
        # for each tensor in batch, zero out values with probability p
        x.div_(keep_prob).mul_(mask)
    return x

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res

def channel_norm(dataset, channel_dim=1)->tuple:
    # collect tensors in list
    l = [data for data, *_ in dataset]
    # join back all tensors so the first dimension is count of tensors
    l = torch.stack(l, dim=0) #size: [N, X, Y, ...] or [N, C, X, Y, ...]

    if channel_dim is None:
        # add redundant first dim
        l = l.unsqueeze(0)

    else:
        # swap channel dimension to first
        l = torch.transpose(l, 0, channel_dim).contiguous() #size: [C, N, X, Y, ...]
    # collapse all except first dimension
    l = l.view(l.size(0), -1) #size: [C, N*X*Y]
    mean = torch.mean(l, dim=1) #size: [C]
    std = torch.std(l, dim=1) #size: [C]
    return mean, std

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            obj.cpu()
    gc.collect()
    torch.cuda.empty_cache()

def memory_allocated(device=None, max=False)->float:
    """Returns allocated memory on device in MBs"""
    if device:
        device = torch.device(device)
        if max:
            alloc = torch.cuda.max_memory_allocated(device=device)
        else:
            alloc = torch.cuda.memory_allocated(device=device)
        alloc /=  1024 ** 2
        return alloc

def print_memory_objects(device=None, max=False)->None:
    numels = Counter()
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())
            numels[obj.device] += obj.numel()
    print()
    for device, numel in sorted(numels.items()):
        print('%s: %s elements, %.3f MBs' % (str(device), numel, numel * 4 / 1024 ** 2))
