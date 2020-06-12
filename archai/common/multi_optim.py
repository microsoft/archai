# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Iterator, List, Optional
from collections import UserList

from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from archai.common.utils import zip_eq


class OptimSched:
    """Holds the optimizer and scheduler"""
    def __init__(self, optim:Optimizer, sched:Optional[_LRScheduler],
                 sched_on_epoch:Optional[bool])->None:
        self.optim = optim
        self.sched = sched
        self.sched_on_epoch = sched_on_epoch

class MultiOptim:
    def __init__(self) -> None:
        self._optim_scheds:List[OptimSched] = []

    def append(self, optim_sched:OptimSched)->None:
        self._optim_scheds.append(optim_sched)

    def zero_grad(self)->None:
        for optim_sched in self._optim_scheds:
            optim_sched.optim.zero_grad()

    def step(self)->None:
        for optim_sched in self._optim_scheds:
            optim_sched.optim.step()
            if optim_sched.sched and not optim_sched.sched_on_epoch:
                optim_sched.sched.step(epoch=None)

    def epoch(self, epoch:Optional[int]=None)->None:
        for optim_sched in self._optim_scheds:
            if optim_sched.sched and optim_sched.sched_on_epoch:
                optim_sched.sched.step(epoch=epoch)

    def get_lr(self, optim_index:int, param_index:int)->float:
        return self._optim_scheds[optim_index].optim.param_groups[param_index]['lr']

    def state_dict(self)->dict:
        optim_states = [optim_sched.optim.state_dict() for optim_sched in self]
        sched_states = [optim_sched.sched.state_dict() if optim_sched.sched else None \
                        for optim_sched in self]

        return {'optim_states': optim_states, 'sched_states':sched_states}

    def load_state_dict(self, state_dict:dict)->None:
        optim_states = state_dict['optim_states']
        sched_states = state_dict['sched_states']

        for optim_sched, optim_state, sched_state in zip_eq(self, optim_states, sched_states):
            optim_sched.optim.load_state_dict(optim_state)
            if optim_sched.sched:
                assert sched_state is not None
                optim_sched.sched.load_state_dict(sched_state)
            else:
                assert sched_state is None

    def __getitem__(self, index)->OptimSched:
        return self._optim_scheds[index]

    def __len__(self)->int:
        return len(self._optim_scheds)

    def __iter__(self)->Iterator[OptimSched]:
        return iter(self._optim_scheds)

