# Copyright (c) 2020 abhuse.
# Licensed under the MIT license.
# https://github.com/ildoonet/pytorch-gradual-warmup-lr

from typing import Any, Dict, List, Optional

from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.optim.optimizer import Optimizer


class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up (increasing) learning rate in optimizer.

    It has been proposed in `Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour`.

    """

    def __init__(
        self, optimizer: Optimizer, multiplier: float, total_epoch: int, after_scheduler: Optional[_LRScheduler] = None
    ) -> None:
        """Initialize the scheduler.

        Args:
            optimizer: Wrapped optimizer.
            multiplier: Target learning rate = base lr * multiplier if multiplier > 1.0.
                If multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
            total_epoch: Target learning rate is reached gradually at total_epoch.
            after_scheduler: After target_epoch, use this scheduler.

        """

        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("Multiplier should be >= 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False

        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Get learning rate at current step.

        Returns:
            List of learning rate at current step.

        """

        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]

    def _step_reduce_lr(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Core computation of step for ReduceLROnPlateau scheduler.

        Args:
            epoch: Current epoch.
            metrics: Metric to monitor.

        """

        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch if epoch != 0 else 1

        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(None, metrics)
            else:
                self.after_scheduler.step(epoch - self.total_epoch, metrics)

    def step(self, epoch: Optional[int] = None, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Step for ReduceLROnPlateau scheduler.

        Args:
            epoch: Current epoch.
            metrics: Metric to monitor.

        """

        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self._step_reduce_lr(epoch, metrics)
