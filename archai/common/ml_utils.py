from typing import Iterable, Type, MutableMapping, Mapping, Any, Optional, Tuple, List, Sequence
import  numpy as np
import logging
import csv
from collections import OrderedDict
import sys
import  os
import pathlib
import time

import  torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.optim import lr_scheduler, SGD, Adam
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _WeightedLoss, _Loss
import torch.nn.functional as F
from torchvision.datasets.utils import check_integrity, download_url
from torch.utils.model_zoo import tqdm

import yaml
import runstats

from .config import Config
from .cocob import CocobBackprop
from torch.utils.data.dataloader import DataLoader


def create_optimizer(conf_opt:Config, params)->Optimizer:
    if conf_opt['type'] == 'sgd':
        return SGD(
           params,
            lr=conf_opt['lr'],
            momentum=conf_opt['momentum'],
            weight_decay=conf_opt['decay'],
            nesterov=conf_opt['nesterov']
        )
    elif conf_opt['type'] == 'adam':
         return Adam(params,
            lr=conf_opt['lr'],
            betas=conf_opt['betas'],
            weight_decay=conf_opt['decay'])
    elif conf_opt['type'] == 'cocob':
        return CocobBackprop(params,
            alpha=conf_opt['alpha'])
    else:
        raise ValueError('invalid optimizer type=%s' % conf_opt['type'])

def get_optim_lr(optimizer:Optimizer)->float:
    for param_group in optimizer.param_groups:
        return param_group['lr']
    raise RuntimeError('optimizer did not had any param_group named lr!')

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


def create_lr_scheduler(conf_lrs:Config, epochs:int, optimizer:Optimizer,
        steps_per_epoch:Optional[int])-> Tuple[Optional[_LRScheduler], bool]:

    # epoch_or_step - apply every epoch or every step
    scheduler, epoch_or_step = None, True # by default sched step on epoch

    # TODO: adjust max epochs for warmup?
    # if conf_lrs.get('warmup', None):
    #     epochs -= conf_lrs['warmup']['epochs']

    if conf_lrs is not None:
        lr_scheduler_type = conf_lrs['type'] # TODO: default should be none?

        if lr_scheduler_type == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                eta_min=conf_lrs['min_lr'])
        elif lr_scheduler_type == 'multi_step':
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=conf_lrs['milestones'],
                                                 gamma=conf_lrs['gamma'])
        elif lr_scheduler_type == 'pyramid':
            scheduler = _adjust_learning_rate_pyramid(optimizer, epochs,
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
                            epochs=epochs, steps_per_epoch=steps_per_epoch,
                        )  # TODO: other params
        elif not lr_scheduler_type:
                scheduler = None
        else:
            raise ValueError('invalid lr_schduler=%s' % lr_scheduler_type)

        # select warmup for LR schedule
        if conf_lrs.get('warmup', None):
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=conf_lrs['warmup']['multiplier'],
                total_epoch=conf_lrs['warmup']['epochs'],
                after_scheduler=scheduler
            )

    return scheduler, epoch_or_step

def _adjust_learning_rate_pyramid(optimizer, max_epoch:int, base_lr:float):
    def _internal_adjust_learning_rate_pyramid(epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = base_lr * (0.1 ** (epoch // (max_epoch * 0.5))) * (0.1 ** (epoch // (max_epoch * 0.75)))
        return lr

    return lr_scheduler.LambdaLR(optimizer, _internal_adjust_learning_rate_pyramid)


# TODO: replace this with SmoothCrossEntropyLoss class
# def cross_entropy_smooth(input: torch.Tensor, target, size_average=True, label_smoothing=0.1):
#     y = torch.eye(10).to(input.device)
#     lb_oh = y[target]

#     target = lb_oh * (1 - label_smoothing) + 0.5 * label_smoothing

#     logsoftmax = nn.LogSoftmax()
#     if size_average:
#         return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
#     else:
#         return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))
class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            # For label smoothing, we replace 1-hot vector with 0.9-hot vector instead.
            # Create empty vector of same size as targets, fill them up with smoothing/(n-1)
            # then replace element where 1 supposed to go and put there 1-smoothing instead
            targets = torch.empty(size=(targets.size(0), n_classes), device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None: # to support weighted targets
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

def get_lossfn(conf_lossfn:Config)->_Loss:
    type = conf_lossfn['type']
    if type == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif type == 'CrossEntropyLabelSmooth':
        return SmoothCrossEntropyLoss(smoothing=conf_lossfn['smoothing'])
    else:
        raise ValueError('criterian type "{}" is not supported'.format(type))

def param_size(module:nn.Module):
    """count all parameters excluding auxiliary"""
    return np.sum(v.numel() for name, v in module.named_parameters() \
        if "auxiliary" not in name)


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
        correct_k = correct[:k].view(-1).float().sum(0)
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