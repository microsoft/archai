# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import copy
from typing import List, Mapping, Optional, Tuple
import pathlib
import math
import statistics

from collections import defaultdict
from torch import Tensor

import yaml

from . import utils, ml_utils
from .common import logger, get_tb_writer
from .apex_utils import ApexUtils


class Metrics:
    """Record top1, top5, loss metrics, track best so far.

    There are 3 levels of metrics:
    1. Run level - these for the one call of 'fit', example, best top1
    2. Epoch level - these are the averages maintained top1, top5, loss
    3. Step level - these are for every step in epoch

    The pre_run must be called before fit call which will reset all metrics. Similarly
    pre_epoch will reset running averages and pre_step will reset step level metrics like average step time.

    The post_step will simply update the running averages while post_epoch updates
    best we have seen for each epoch.
    """

    def __init__(self, title:str, apex:Optional[ApexUtils], logger_freq:int=50) -> None:
        """Create the metrics object to maintain epoch stats

        Arguments:
            title {str} -- descriptive name of the stage for which metrics are collected
        Keyword Arguments:
            logger_freq {int} -- Must be > 0 for epoch level logging, the step level logging is decided by this number (default: {50})
        """
        self.logger_freq = logger_freq
        self.title = title
        self._apex = apex
        self._reset_run()

    def _reset_run(self)->None:
        self.run_metrics = RunMetrics()
        self.global_step = -1
        self._tb_path = logger.path()

    def pre_run(self)->None:
        self._reset_run()
        self.run_metrics.pre_run()

    def post_run(self, test_metrics:Optional['Metrics']=None)->None:
        self.run_metrics.post_run(test_metrics)

        # logging
        if self.logger_freq > 0:
            with logger.pushd('timings'):
                logger.info({'epoch':self.run_metrics.epoch_time_avg(),
                            'step': self.run_metrics.step_time_avg(),
                            'run': self.run_metrics.duration()})
                if self.is_dist():
                    logger.info({'dist_epoch_sum': self.reduce_sum(self.run_metrics.epoch_time_avg()),
                                'dist_step': self.reduce_mean(self.run_metrics.step_time_avg()),
                                'dist_run_sum': self.reduce_sum(self.run_metrics.duration())})


            best_train, best_val, best_test = self.run_metrics.best_epoch()
            with logger.pushd('best_train'):
                logger.info({'epoch': best_train.index,
                            'top1': best_train.top1.avg})
                if self.is_dist():
                    logger.info({'dist_epoch': self.reduce_mean(best_train.index),
                                'dist_top1': self.reduce_mean(best_train.top1.avg)})

            if best_val:
                with logger.pushd('best_val'):
                    logger.info({'epoch': best_val.index,
                                'top1': best_val.top1.avg})
                    if self.is_dist():
                        logger.info({'dist_epoch': self.reduce_mean(best_val.index),
                                    'dist_top1': self.reduce_mean(best_val.top1.avg)})

            if best_test:
                with logger.pushd('best_test'):
                    logger.info({'epoch': best_test.index,
                                'top1': best_test.top1.avg})
                    if self.is_dist():
                        logger.info({'dist_epoch': self.reduce_mean(best_test.index),
                                    'dist_top1': self.reduce_mean(best_test.top1.avg)})

    def pre_step(self, x: Tensor, y: Tensor):
        self.run_metrics.cur_epoch().pre_step()
        self.global_step += 1

    def post_step(self, x: Tensor, y: Tensor, logits: Tensor,
                  loss: Tensor, steps: int) -> None:
        assert len(x)==len(y) and len(y)==len(logits) and len(loss.shape)==0
        # update metrics after optimizer step
        batch_size = x.size(0)
        top1, top5 = ml_utils.accuracy(logits, y, topk=(1, 5))

        epoch = self.run_metrics.cur_epoch()
        epoch.post_step(top1.item(), top5.item(),
                        loss.item(), batch_size)

        if self.logger_freq > 0 and \
                ((epoch.step+1) % self.logger_freq == 0):
            logger.info({'top1': epoch.top1.avg,
                        'top5': epoch.top5.avg,
                        'loss': epoch.loss.avg,
                        'step_time': epoch.step_time.last})

            if self.is_dist():
                logger.info({'dist_top1': self.reduce_mean(epoch.top1.avg),
                            'dist_top5': self.reduce_mean(epoch.top5.avg),
                            'dist_loss': self.reduce_mean(epoch.loss.avg),
                            'dist_step_time': self.reduce_mean(epoch.step_time.last)})


        # NOTE: Tensorboard step-level logging is removed as it becomes exponentially expensive on Azure blobs
        # writer = get_tb_writer()
        # writer.add_scalar(f'{self._tb_path}/train_steps/loss',
        #                     epoch.loss.avg, self.global_step)
        # writer.add_scalar(f'{self._tb_path}/train_steps/top1',
        #                     epoch.top1.avg, self.global_step)
        # writer.add_scalar(f'{self._tb_path}/train_steps/top5',
        #                     epoch.top5.avg, self.global_step)

    def pre_epoch(self, lr:float=math.nan)->None:
        epoch = self.run_metrics.add_epoch()
        epoch.pre_epoch(lr)
        if lr is not None:
            writer = get_tb_writer()
            if writer is not None:
                if self.logger_freq > 0 and not math.isnan(lr):
                    logger.debug({'start_lr': lr})
                writer.add_scalar(f'{self._tb_path}/train_steps/lr',
                                    lr, self.global_step)

    def post_epoch(self, lr:float=math.nan, val_metrics:Optional['Metrics']=None):
        epoch = self.run_metrics.cur_epoch()
        epoch.post_epoch(lr, val_metrics)

        val_epoch_metrics = None
        if val_metrics:
            val_epoch_metrics = val_metrics.run_metrics.epochs_metrics[-1]

        if self.logger_freq > 0:
            with logger.pushd('train'):
                logger.info({'top1': epoch.top1.avg,
                            'top5': epoch.top5.avg,
                            'loss': epoch.loss.avg,
                            'duration': epoch.duration(),
                            'step_time': epoch.step_time.avg,
                            'end_lr': lr})
                if self.is_dist():
                    logger.info({'dist_top1': self.reduce_mean(epoch.top1.avg),
                                'dist_top5': self.reduce_mean(epoch.top5.avg),
                                'dist_loss': self.reduce_mean(epoch.loss.avg),
                                'dist_duration': self.reduce_mean(epoch.duration()),
                                'dist_step_time': self.reduce_mean(epoch.step_time.avg),
                                'dist_end_lr': self.reduce_mean(lr)})
            if val_epoch_metrics:
                with logger.pushd('val'):
                    logger.info({'top1': val_epoch_metrics.top1.avg,
                                'top5': val_epoch_metrics.top5.avg,
                                'loss': val_epoch_metrics.loss.avg,
                                'duration': val_epoch_metrics.duration()})
                    if self.is_dist():
                        logger.info({'dist_top1': self.reduce_mean(val_epoch_metrics.top1.avg),
                                    'dist_top5': self.reduce_mean(val_epoch_metrics.top5.avg),
                                    'dist_loss': self.reduce_mean(val_epoch_metrics.loss.avg),
                                    'dist_duration': self.reduce_mean(val_epoch_metrics.duration())})

        # writer = get_tb_writer()
        # writer.add_scalar(f'{self._tb_path}/train_epochs/loss',
        #                     epoch.loss.avg, epoch.index)
        # writer.add_scalar(f'{self._tb_path}/train_epochs/top1',
        #                     epoch.top1.avg, epoch.index)
        # writer.add_scalar(f'{self._tb_path}/train_epochs/top5',
        #                     epoch.top5.avg, epoch.index)
        # if test_epoch:
        #     writer.add_scalar(f'{self._tb_path}/val_epochs/loss',
        #                         test_epoch.loss.avg, epoch.index)
        #     writer.add_scalar(f'{self._tb_path}/val_epochs/top1',
        #                         test_epoch.top1.avg, epoch.index)
        #     writer.add_scalar(f'{self._tb_path}/val_epochs/top5',
        #                         test_epoch.top5.avg, epoch.index)

    def state_dict(self)->Mapping:
        return utils.state_dict(self)

    def load_state_dict(self, state_dict:dict)->None:
        utils.load_state_dict(self, state_dict)

    def __getstate__(self):
        state = self.__dict__.copy()
        if '_apex' in state:
            del state['_apex'] # cannot serialize this
        return state
    # no need to define __setstate__ because _apex should be set from constructor

    def save(self, filepath:str)->Optional[str]:
        if filepath:
            filepath = utils.full_path(filepath)
            pathlib.Path(filepath).write_text(yaml.dump(self))
        return filepath

    def epochs(self)->int:
        """Returns epochs recorded so far"""
        return len(self.run_metrics.epochs_metrics)

    def cur_epoch(self)->'EpochMetrics':
        return self.run_metrics.cur_epoch()

    def reduce_min(self, val):
        if not self._apex:
            return val
        return self._apex.reduce(val, op='min')
    def reduce_max(self, val):
        if not self._apex:
            return val
        return self._apex.reduce(val, op='max')
    def reduce_sum(self, val):
        if not self._apex:
            return val
        return self._apex.reduce(val, op='sum')
    def reduce_mean(self, val):
        if not self._apex:
            return val
        return self._apex.reduce(val, op='mean')
    def is_dist(self)->bool:
        if not self._apex:
            return False
        return self._apex.is_dist()

    def best_train_top1(self)->float:
        return self.run_metrics.best_epoch()[0].top1.avg
    def best_val_top1(self)->float:
        val_epoch_metrics = self.run_metrics.best_epoch()[1]
        return val_epoch_metrics.top1.avg if val_epoch_metrics is not None else math.nan
    def best_test_top1(self)->float:
        test_epoch_metrics = self.run_metrics.best_epoch()[2]
        return test_epoch_metrics.top1.avg if test_epoch_metrics is not None else math.nan

class Accumulator:
    # TODO: replace this with Metrics class
    def __init__(self):
        self.metrics = defaultdict(lambda: 0.)

    def add(self, key, value):
        self.metrics[key] += value

    def add_dict(self, dict):
        for key, value in dict.items():
            self.add(key, value)

    def __getitem__(self, item):
        return self.metrics[item]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def get_dict(self):
        return copy.deepcopy(dict(self.metrics))

    def items(self):
        return self.metrics.items()

    def __str__(self):
        return str(dict(self.metrics))

    def __truediv__(self, other):
        newone = Accumulator()
        for key, value in self.items():
            if isinstance(other, str):
                if other != key:
                    newone[key] = value / self[other]
                else:
                    newone[key] = value
            else:
                newone[key] = value / other
        return newone

class EpochMetrics:
    """Stores the metrics for each epoch. Training metrics is in top1, top5 etc
    while validation metrics is in val_metrics"""
    def __init__(self, index:int) -> None:
        self.index = index
        self.top1 = utils.AverageMeter()
        self.top5 = utils.AverageMeter()
        self.loss = utils.AverageMeter()
        self.step_time = utils.AverageMeter()
        self.start_time = math.nan
        self.end_time = math.nan
        self.step = -1
        self.start_lr = math.nan
        self.end_lr = math.nan
        self.val_metrics:Optional[EpochMetrics] = None

    def pre_step(self):
        self._step_start_time = time.time()
        self.step += 1
    def post_step(self, top1:float, top5:float, loss:float, batch:int):
        self.step_time.update(time.time() - self._step_start_time)
        self.top1.update(top1, batch)
        self.top5.update(top5, batch)
        self.loss.update(loss, batch)

    def pre_epoch(self, lr:float):
        self.start_time = time.time()
        self.start_lr = lr
    def post_epoch(self, lr:float, val_metrics:Optional[Metrics]):
        self.end_time = time.time()
        self.end_lr = lr

        if val_metrics is not None:
            assert len(val_metrics.run_metrics.epochs_metrics)==1, 'Number of epochs in val metrics should be 1'
            self.val_metrics = val_metrics.run_metrics.epochs_metrics[-1]

    def duration(self):
        return self.end_time-self.start_time

class RunMetrics:
    """Metrics for the entire run. It mainly consist of metrics for each epoch"""
    def __init__(self) -> None:
        self.epochs_metrics:List[EpochMetrics] = []
        self.start_time = math.nan
        self.end_time = math.nan
        self.epoch = -1
        self.test_metrics:Optional['Metrics'] = None

    def pre_run(self):
        self.start_time = time.time()
    def post_run(self, test_metrics:Optional['Metrics']=None):
        self.end_time = time.time()
        self.test_metrics = test_metrics
        # test should have only one epoch
        assert test_metrics is None or len(test_metrics.run_metrics.epochs_metrics)==1

    def add_epoch(self)->EpochMetrics:
        self.epoch = len(self.epochs_metrics)
        epoch_metrics = EpochMetrics(self.epoch)
        self.epochs_metrics.append(epoch_metrics)
        return epoch_metrics

    def cur_epoch(self)->EpochMetrics:
        return self.epochs_metrics[self.epoch]

    def best_epoch(self)->Tuple[EpochMetrics, Optional[EpochMetrics],
                                Optional[EpochMetrics]]: # [train, val, test]
        best_train = max(self.epochs_metrics, key=lambda e:e.top1.avg)

        best_val = max(self.epochs_metrics,
            key=lambda e:e.val_metrics.top1.avg if e.val_metrics else -1)
        best_val = best_val.val_metrics if best_val.val_metrics else None

        best_test = self.test_metrics.run_metrics.epochs_metrics[-1] \
                    if self.test_metrics else None

        return best_train, best_val, best_test

    def epoch_time_avg(self):
        return statistics.mean((e.duration() for e in self.epochs_metrics))
    def step_time_avg(self):
        return statistics.mean((e.step_time.avg for e in self.epochs_metrics))

    def duration(self):
        return self.end_time-self.start_time