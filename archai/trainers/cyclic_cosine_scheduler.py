# Copyright (c) 2019 abhuse.
# Licensed under the MIT license.
# https://github.com/abhuse/cyclic-cosine-decay/blob/master/scheduler.py

from collections.abc import Iterable
from math import cos, floor, log, pi
from typing import List, Optional, Union

import torch
from torch.optim.lr_scheduler import _LRScheduler


class CyclicCosineDecayLR(_LRScheduler):
    """A learning rate scheduler for cyclic cosine annealing.

    This scheduler is useful when doing QAT provinding
    a ~0.3 ppl boost over the traditional cosine annealing scheduler.

    For more details and code, see the project's GitHub repository:
    https://github.com/abhuse/cyclic-cosine-decay

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        init_decay_epochs: int,
        min_decay_lr: Union[float, List[float]],
        restart_interval: Optional[int] = None,
        restart_interval_multiplier: Optional[float] = None,
        restart_lr: Optional[Union[float, List[float]]] = None,
        warmup_epochs: Optional[int] = None,
        warmup_start_lr: Optional[Union[float, List[float]]] = None,
        last_epoch: Optional[int] = -1,
        verbose: Optional[bool] = False,
    ) -> None:
        """Override the initialization of `_LRScheduler` with custom attributes.

        Args:
            optimizer: The optimizer to use. This should be an instance of
                `torch.optim.Optimizer`.
            init_decay_epochs: The number of epochs for the initial decay period.
            min_decay_lr: The learning rate at the end of the initial decay period.
            restart_interval: The interval between restarts of the cyclic schedule.
                This should be a positive integer, or `None` to disable restarts.
                restart_interval_multiplier: The coefficient used to increase the
                restart interval between cycles. This should be a positive float,
                or `None` to use a fixed restart interval.
            restart_lr: The learning rate at the start of a cycle. This should be
                a positive float or a list of floats with the same length as
                `optimizer.param_groups`, or `None` to use the current learning
                rates of the optimizer.
            warmup_epochs: The number of epochs to use for a warmup period. This
                should be a positive integer, or `None` to disable warmup.
            warmup_start_lr: The learning rate at the start of the warmup period.
                This should be a positive float or a list of floats with the same
                length as `optimizer.param_groups`, and must be set if `warmup_epochs`
                is not `None`.
            last_epoch: The index of the last epoch. This is used when resuming a
                training job.
            verbose: Whether to print a message to stdout for each update.

        """

        if not isinstance(init_decay_epochs, int) or init_decay_epochs < 1:
            raise ValueError("init_decay_epochs must be positive integer, got {} instead".format(init_decay_epochs))

        if isinstance(min_decay_lr, Iterable) and len(min_decay_lr) != len(optimizer.param_groups):
            raise ValueError(
                "Expected len(min_decay_lr) to be equal to len(optimizer.param_groups), "
                "got {} and {} instead".format(len(min_decay_lr), len(optimizer.param_groups))
            )

        if restart_interval is not None and (not isinstance(restart_interval, int) or restart_interval < 1):
            raise ValueError("restart_interval must be positive integer, got {} instead".format(restart_interval))

        if restart_interval_multiplier is not None and (
            not isinstance(restart_interval_multiplier, float) or restart_interval_multiplier <= 0
        ):
            raise ValueError(
                "restart_interval_multiplier must be positive float, got {} instead".format(restart_interval_multiplier)
            )

        if isinstance(restart_lr, Iterable) and len(restart_lr) != len(optimizer.param_groups):
            raise ValueError(
                "Expected len(restart_lr) to be equal to len(optimizer.param_groups), "
                "got {} and {} instead".format(len(restart_lr), len(optimizer.param_groups))
            )

        if warmup_epochs is not None:
            if not isinstance(warmup_epochs, int) or warmup_epochs < 1:
                raise ValueError(
                    "Expected warmup_epochs to be positive integer, got {} instead".format(type(warmup_epochs))
                )

            if warmup_start_lr is None:
                raise ValueError("warmup_start_lr must be set when warmup_epochs is not None")

            if not (isinstance(warmup_start_lr, float) or isinstance(warmup_start_lr, Iterable)):
                raise ValueError(
                    "warmup_start_lr must be either float or iterable of floats, got {} instead".format(warmup_start_lr)
                )

            if isinstance(warmup_start_lr, Iterable) and len(warmup_start_lr) != len(optimizer.param_groups):
                raise ValueError(
                    "Expected len(warmup_start_lr) to be equal to len(optimizer.param_groups), "
                    "got {} and {} instead".format(len(warmup_start_lr), len(optimizer.param_groups))
                )

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
        """Return the learning rate.

        This is the learning rate that will be used for the next iteration of training.

        Returns:
            The learning rate.

        """

        if self._warmup_epochs > 0 and self.last_epoch < self._warmup_epochs:
            return self._calc(self.last_epoch, self._warmup_epochs, self._warmup_start_lr, self.base_lrs)
        elif self.last_epoch < self._init_decay_epochs + self._warmup_epochs:
            return self._calc(
                self.last_epoch - self._warmup_epochs, self._init_decay_epochs, self.base_lrs, self._min_decay_lr
            )
        else:
            if self._restart_interval is not None:
                if self._restart_interval_multiplier is None:
                    cycle_epoch = (
                        self.last_epoch - self._init_decay_epochs - self._warmup_epochs
                    ) % self._restart_interval
                    lrs = self.base_lrs if self._restart_lr is None else self._restart_lr

                    return self._calc(cycle_epoch, self._restart_interval, lrs, self._min_decay_lr)
                else:
                    n = self._get_n(self.last_epoch - self._warmup_epochs - self._init_decay_epochs)
                    sn_prev = self._partial_sum(n)
                    cycle_epoch = self.last_epoch - sn_prev - self._warmup_epochs - self._init_decay_epochs
                    interval = self._restart_interval * self._restart_interval_multiplier**n
                    lrs = self.base_lrs if self._restart_lr is None else self._restart_lr

                    return self._calc(cycle_epoch, interval, lrs, self._min_decay_lr)
            else:
                return self._min_decay_lr

    def _calc(self, t: int, T: int, lrs: List[float], min_lrs: List[float]) -> List[float]:
        """Calculate the learning rate for the current cycle epoch.

        Args:
            t: The current cycle epoch.
            T: The total number of epochs in the current cycle.
            lrs: The initial learning rates for each parameter group.
            min_lrs: The minimum learning rates for each parameter group.

        Returns:
            The annealed learning rates for each parameter group.

        """

        return [min_lr + (lr - min_lr) * ((1 + cos(pi * t / T)) / 2) for lr, min_lr in zip(lrs, min_lrs)]

    def _get_n(self, epoch: int) -> int:
        """Return the value of `n` for the current epoch.

        Args:
            epoch: The current epoch.

        Returns:
            int: The value of `n` for the current epoch.

        """

        _t = 1 - (1 - self._restart_interval_multiplier) * epoch / self._restart_interval

        return floor(log(_t, self._restart_interval_multiplier))

    def _partial_sum(self, n: int) -> float:
        """Calculate the partial sum of the geometric sequence.

        Args:
            n: The exponent of the current epoch.

        Returns:
            The partial sum of the geometric sequence.

        """

        return (
            self._restart_interval
            * (1 - self._restart_interval_multiplier**n)
            / (1 - self._restart_interval_multiplier)
        )
