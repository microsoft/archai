# MIT License
#
# Copyright (c) 2019 abhuse
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Cyclic Cosine learning rate annealing.
"""

from math import cos, floor, log, pi
from typing import Iterable, Optional, Union

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class CyclicCosineDecayLR(_LRScheduler):
    """Implements the Cyclic Cosine annealing.

    This learning rate scheduler is useful when doing QAT provinding
    a ~0.3 ppl boost over the traditional cosine annealing scheduler.

    Code and additional documentation: https://github.com/abhuse/cyclic-cosine-decay.

    """

    def __init__(self,
                 optimizer: Optimizer,
                 init_decay_epochs: int,
                 min_decay_lr: Union[float, Iterable[float]],
                 restart_interval: Optional[int] = None,
                 restart_interval_multiplier: Optional[float] = None,
                 restart_lr: Optional[Union[float, Iterable[float]]] = None,
                 warmup_epochs: Optional[int] = None,
                 warmup_start_lr: Optional[Union[float, Iterable[float]]] = None,
                 last_epoch: Optional[int] = -1,
                 verbose: Optional[bool] = False) -> None:
        """Initializes a new CyclicCosineDecayLR object.

        Args:
            optimizer: Wrapped optimizer.
            init_decay_epochs: Number of initial decay epochs.
            min_decay_lr: Learning rate at the end of decay.
            restart_interval: Restart interval for fixed cycles.
            restart_interval_multiplier: Multiplication coefficient for geometrically increasing cycles.
            restart_lr: Learning rate when cycle restarts.
            warmup_epochs: Number of warmup epochs.
            warmup_start_lr: Learning rate at the beginning of warmup.
            last_epoch: The index of the last epoch. This parameter is used when resuming a training job.
            verbose: If true, prints a message to stdout for each update.

        """

        if not isinstance(init_decay_epochs, int) or init_decay_epochs < 1:
            raise ValueError("init_decay_epochs must be positive integer, got {} instead".format(init_decay_epochs))

        if isinstance(min_decay_lr, Iterable) and len(min_decay_lr) != len(optimizer.param_groups):
            raise ValueError("Expected len(min_decay_lr) to be equal to len(optimizer.param_groups), "
                             "got {} and {} instead".format(len(min_decay_lr), len(optimizer.param_groups)))

        if restart_interval is not None and (not isinstance(restart_interval, int) or restart_interval < 1):
            raise ValueError("restart_interval must be positive integer, got {} instead".format(restart_interval))

        if restart_interval_multiplier is not None and \
                (not isinstance(restart_interval_multiplier, float) or restart_interval_multiplier <= 0):
            raise ValueError("restart_interval_multiplier must be positive float, got {} instead".format(
                restart_interval_multiplier))

        if isinstance(restart_lr, Iterable) and len(restart_lr) != len(optimizer.param_groups):
            raise ValueError("Expected len(restart_lr) to be equal to len(optimizer.param_groups), "
                             "got {} and {} instead".format(len(restart_lr), len(optimizer.param_groups)))

        if warmup_epochs is not None:
            if not isinstance(warmup_epochs, int) or warmup_epochs < 1:
                raise ValueError(
                    "Expected warmup_epochs to be positive integer, got {} instead".format(type(warmup_epochs)))

            if warmup_start_lr is None:
                raise ValueError("warmup_start_lr must be set when warmup_epochs is not None")

            if not (isinstance(warmup_start_lr, float) or isinstance(warmup_start_lr, Iterable)):
                raise ValueError("warmup_start_lr must be either float or iterable of floats, got {} instead".format(
                    warmup_start_lr))

            if isinstance(warmup_start_lr, Iterable) and len(warmup_start_lr) != len(optimizer.param_groups):
                raise ValueError("Expected len(warmup_start_lr) to be equal to len(optimizer.param_groups), "
                                 "got {} and {} instead".format(len(warmup_start_lr), len(optimizer.param_groups)))

        group_num = len(optimizer.param_groups)

        self._warmup_start_lr = [warmup_start_lr] * group_num if isinstance(warmup_start_lr, float) else warmup_start_lr
        self._warmup_epochs = 0 if warmup_epochs is None else warmup_epochs
        self._init_decay_epochs = init_decay_epochs
        self._min_decay_lr = [min_decay_lr] * group_num if isinstance(min_decay_lr, float) else min_decay_lr
        self._restart_lr = [restart_lr] * group_num if isinstance(restart_lr, float) else restart_lr
        self._restart_interval = restart_interval
        self._restart_interval_multiplier = restart_interval_multiplier

        super(CyclicCosineDecayLR, self).__init__(optimizer, last_epoch, verbose=verbose)

    def get_lr(self) -> float:
        """Gets current learning rate.

        Returns:
            (float): Learning rate.

        """

        if self._warmup_epochs > 0 and self.last_epoch < self._warmup_epochs:
            return self._calc(self.last_epoch,
                              self._warmup_epochs,
                              self._warmup_start_lr,
                              self.base_lrs)

        elif self.last_epoch < self._init_decay_epochs + self._warmup_epochs:
            return self._calc(self.last_epoch - self._warmup_epochs,
                              self._init_decay_epochs,
                              self.base_lrs,
                              self._min_decay_lr)

        else:
            if self._restart_interval is not None:
                if self._restart_interval_multiplier is None:
                    cycle_epoch = (self.last_epoch - self._init_decay_epochs - self._warmup_epochs) % self._restart_interval
                    lrs = self.base_lrs if self._restart_lr is None else self._restart_lr

                    return self._calc(cycle_epoch,
                                      self._restart_interval,
                                      lrs,
                                      self._min_decay_lr)

                else:
                    n = self._get_n(self.last_epoch - self._warmup_epochs - self._init_decay_epochs)
                    sn_prev = self._partial_sum(n)
                    cycle_epoch = self.last_epoch - sn_prev - self._warmup_epochs - self._init_decay_epochs
                    interval = self._restart_interval * self._restart_interval_multiplier ** n
                    lrs = self.base_lrs if self._restart_lr is None else self._restart_lr

                    return self._calc(cycle_epoch,
                                      interval,
                                      lrs,
                                      self._min_decay_lr)

            else:
                return self._min_decay_lr

    def _calc(self,
              t: int,
              T: int,
              lrs: Union[float, Iterable[float]],
              min_lrs: Union[float, Iterable[float]]) -> float:
        """Calculates the current learning rate according to schedule.

        Args:
            t: Current timestep.
            T: Maximum timesteps.
            lrs: Learning rates.
            min_lrs: Minimum learning rates.

        Returns:
            (float): Learning rate.

        """

        return [min_lr + (lr - min_lr) * ((1 + cos(pi * t / T)) / 2)
                for lr, min_lr in zip(lrs, min_lrs)]

    def _get_n(self, epoch: int) -> float:
        """Gets the number of current timestep.

        Args:
            epoch: Epoch value.
        
        Returns:
            (float): Current timestep.

        """

        _t = 1 - (1 - self._restart_interval_multiplier) * epoch / self._restart_interval

        return floor(log(_t, self._restart_interval_multiplier))

    def _partial_sum(self, n: int) -> float:
        """Calculates the partial sum.

        Args:
            n: Number of items to be summed.

        Returns:
            (float): Partial sum.

        """

        return self._restart_interval * (1 - self._restart_interval_multiplier ** n) / (
                    1 - self._restart_interval_multiplier)
