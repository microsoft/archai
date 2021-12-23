# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License
#
# Copyright (c) 2019 cybertronai
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

"""Lamb-based optimization algorithm.
"""

from typing import Iterable, Optional, Tuple

import torch
from torch.optim import Optimizer


class Lamb(Optimizer):
    """Implements the Lamb algorithm.

    Large Batch Optimization for Deep Learning: Training BERT in 76 minutes
    https://arxiv.org/abs/1904.00962

    """

    def __init__(self,
                 params: Iterable,
                 lr: Optional[float] = 1e-3,
                 betas: Optional[Tuple[float]] = (0.9, 0.999),
                 eps: Optional[float] = 1e-6,
                 weight_decay: Optional[float] = 0.0,
                 adam: Optional[bool] = False) -> None:
        """Overrides the initialization method.

        Args:
            params: Iterable of parameters.
            lr: Learning rate.
            betas: Coefficients used for computing running averages.
            eps: Term added to the denominator to improve numerical stability.
            weight_decay: Weight decay.
            adam: Whether to turn current optimizer into Adam.

        """

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        self.adam = adam
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure: Optional[callable] = None) -> torch.FloatTensor:
        """Performs a single optimization step.

        Args:
            closure: Reevaluates the model and returns the loss.

        Returns:
            (torch.TensorFloat): Loss value.
            
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Paper v3 does not use debiasing.
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast.
                step_size = group['lr'] # * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.data.norm(p=2).clamp_(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(group['weight_decay'], p.data)

                adam_norm = adam_step.norm(p=2)

                if weight_norm == 0.0 or adam_norm == 0.0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / (adam_norm + group['eps'])

                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(-step_size * trust_ratio, adam_step)

        return loss


@torch.jit.script
def lamb_kernel(param: torch.Tensor,
                grad: torch.Tensor,
                exp_avg: torch.Tensor,
                exp_avg_sq: torch.Tensor,
                beta1: float,
                beta2: float,
                step_size: float,
                eps: float,
                weight_decay: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

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
    """Implements the JIT-based Lamb algorithm.

    Large Batch Optimization for Deep Learning: Training BERT in 76 minutes
    https://arxiv.org/abs/1904.00962

    """

    def __init__(self,
                 params: Iterable,
                 lr: Optional[float] = 1e-3,
                 betas: Optional[Tuple[float]] = (0.9, 0.999),
                 eps: Optional[float] = 1e-6,
                 weight_decay: Optional[float] = 0.0,
                 adam: Optional[bool] = False) -> None:
        """Overrides the initialization method.

        Args:
            params: Iterable of parameters.
            lr: Learning rate.
            betas: Coefficients used for computing running averages.
            eps: Term added to the denominator to improve numerical stability.
            weight_decay: Weight decay.
            adam: Whether to turn current optimizer into Adam.

        """

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        self.adam = adam
        super().__init__(params, defaults)

    def step(self, closure: Optional[callable] = None) -> torch.FloatTensor:
        """Performs a single optimization step.

        Args:
            closure: Reevaluates the model and returns the loss.

        Returns:
            (torch.TensorFloat): Loss value.
            
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                step_size = group['lr']

                param, exp_avg, exp_avg_sq = lamb_kernel(p.data, grad, exp_avg,
                                                         exp_avg_sq, beta1,
                                                         beta2, step_size,
                                                         group['eps'],
                                                         group['weight_decay'],
                                                         )
                state['exp_avg'] = exp_avg
                state['exp_avg_sq'] = exp_avg_sq
                p.data = param

        return loss
