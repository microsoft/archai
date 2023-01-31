# Copyright (c) 2019-2020, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0.
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py
#
# Copyright (c) 2019 cybertronai.
# Licensed under the MIT license.

from typing import Iterable, Optional, Tuple

import torch
from torch.optim import Optimizer


class Lamb(Optimizer):
    """Lamb algorithm for large batch optimization.

    It has been proposed in `Large Batch Optimization for Deep Learning:
    Training BERT in 76 minutes`.

    Reference:
        https://arxiv.org/abs/1904.00962

    """

    def __init__(
        self,
        params: Iterable,
        lr: Optional[float] = 1e-3,
        betas: Optional[Tuple[float, float]] = (0.9, 0.999),
        eps: Optional[float] = 1e-6,
        weight_decay: Optional[float] = 0.0,
        adam: Optional[bool] = False,
    ) -> None:
        """Initialize the optimizer.

        Args:
            params: An iterable of parameters to optimize.
            lr: The learning rate.
            betas: Coefficients used for computing running averages.
            eps: Term added to the denominator to improve numerical stability.
            weight_decay: Weight decay.
            adam: Whether to turn current optimizer into Adam.

        Raises:
            ValueError: If the learning rate, epsilon value, or beta parameters are invalid.

        """

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        self.adam = adam

        super().__init__(params, defaults)

    def step(self, closure: Optional[callable] = None) -> torch.FloatTensor:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError("Lamb does not support sparse gradients.")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast
                # * math.sqrt(bias_correction2) / bias_correction1
                step_size = group["lr"]

                weight_norm = p.data.norm(p=2).clamp_(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group["eps"])
                if group["weight_decay"] != 0:
                    adam_step.add_(p.data, alpha=group["weight_decay"])

                adam_norm = adam_step.norm(p=2)

                if weight_norm == 0.0 or adam_norm == 0.0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / (adam_norm + group["eps"])

                state["weight_norm"] = weight_norm
                state["adam_norm"] = adam_norm
                state["trust_ratio"] = trust_ratio

                if self.adam:
                    trust_ratio = 1

                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss


@torch.jit.script
def _lamb_kernel(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    beta1: float,
    beta2: float,
    step_size: float,
    eps: float,
    weight_decay: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    exp_avg = exp_avg * beta1 + (1 - beta1) * grad
    exp_avg_sq = exp_avg_sq * beta2 + (1 - beta2) * (grad * grad)

    adam_step = exp_avg / (exp_avg_sq.sqrt() + eps)
    adam_step = adam_step + weight_decay * param

    weight_norm = param.norm(p=2).clamp(0, 10)
    adam_norm = adam_step.norm(p=2)

    trust_ratio = weight_norm / (adam_norm + eps)
    trust_ratio = (weight_norm == 0.0) * 1.0 + (weight_norm != 0.0) * trust_ratio
    trust_ratio = (adam_norm == 0.0) * 1.0 + (adam_norm != 0.0) * trust_ratio
    trust_ratio = trust_ratio.float()

    param = param - step_size * trust_ratio * adam_step

    return param, exp_avg, exp_avg_sq


class JITLamb(Optimizer):
    """JIT-based version of the Lamb algorithm for large batch optimization.

    It has been proposed in `Large Batch Optimization for Deep Learning:
    Training BERT in 76 minutes`.

    Reference:
        https://arxiv.org/abs/1904.00962

    """

    def __init__(
        self,
        params: Iterable,
        lr: Optional[float] = 1e-3,
        betas: Optional[Tuple[float, float]] = (0.9, 0.999),
        eps: Optional[float] = 1e-6,
        weight_decay: Optional[float] = 0.0,
        adam: Optional[bool] = False,
    ) -> None:
        """Initialize the optimizer.

        Args:
            params: An iterable of parameters to optimize.
            lr: The learning rate.
            betas: Coefficients used for computing running averages.
            eps: Term added to the denominator to improve numerical stability.
            weight_decay: Weight decay.
            adam: Whether to turn current optimizer into Adam.

        Raises:
            ValueError: If the learning rate, epsilon value, or beta parameters are invalid.

        """

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        self.adam = adam

        super().__init__(params, defaults)

    def step(self, closure: Optional[callable] = None) -> torch.FloatTensor:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError("JITLamb does not support sparse gradients.")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                step_size = group["lr"]

                param, exp_avg, exp_avg_sq = _lamb_kernel(
                    p.data,
                    grad,
                    exp_avg,
                    exp_avg_sq,
                    beta1,
                    beta2,
                    step_size,
                    group["eps"],
                    group["weight_decay"],
                )
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq
                p.data = param

        return loss
