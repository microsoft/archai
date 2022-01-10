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

"""Log Uniform Sampler.
"""

from typing import Type

import torch
from torch import nn


class LogUniformSampler:
    """Implements a sampler based on the logarithm uniform distribution.

    """

    def __init__(self,
                 range_max: float,
                 n_sample: int) -> None:
        """Overrides initialization method.

        Args:
            range_max: Maximum range of sampler.
            n_sample: Number of samples.

        Reference:
            https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/candidate_sampling_ops.py
            
        """

        with torch.no_grad():
            self.range_max = range_max

            log_indices = torch.arange(1., range_max+2., 1.).log_()

            self.dist = (log_indices[1:] - log_indices[:-1]) / log_indices[-1]
            self.log_q = (- (-self.dist.double().log1p_() * 2 * n_sample).expm1_()).log_().float()

        self.n_sample = n_sample

    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        """Performs the actual sampling.

        Args:
            labels: Labels to be sampled.

        Returns:
            (torch.Tensor): Sampled values.

        """

        n_sample = self.n_sample
        n_tries = 2 * n_sample

        with torch.no_grad():
            device = labels.device

            neg_samples = torch.multinomial(self.dist, n_tries, replacement=True).unique()
            neg_samples = neg_samples.to(device)

            true_log_probs = self.log_q[labels].to(device)
            samp_log_probs = self.log_q[neg_samples].to(device)

            return true_log_probs, samp_log_probs, neg_samples


def sample_logits(embedding: nn.Embedding,
                  bias: torch.Tensor,
                  labels: torch.Tensor,
                  inputs: torch.Tensor,
                  sampler: Type['LogUniformSampler']) -> torch.Tensor:
    """Samples the logits.

    Args:
        embedding: An nn.Embedding instance.
        bias: Bias tensor.
        labels: Labels tensor.
        inputs: Inputs tensor.
        sampler: An instance of a LogUniformSampler.

    Returns:
        (torch.Tensor): Sampled logits.

    """

    true_log_probs, samp_log_probs, neg_samples = sampler.sample(labels)

    n_sample = neg_samples.size(0)

    b1, b2 = labels.size(0), labels.size(1)
    
    all_ids = torch.cat([labels.view(-1), neg_samples])

    all_w = embedding(all_ids)
    true_w = all_w[: -n_sample].view(b1, b2, -1)
    sample_w = all_w[- n_sample:].view(n_sample, -1)

    all_b = bias[all_ids]
    true_b = all_b[: -n_sample].view(b1, b2)
    sample_b = all_b[- n_sample:]

    hit = (labels[:, :, None] == neg_samples).detach()

    true_logits = torch.einsum('ijk,ijk->ij',
        [true_w, inputs]) + true_b - true_log_probs
    sample_logits = torch.einsum('lk,ijk->ijl',
        [sample_w, inputs]) + sample_b - samp_log_probs
    sample_logits.masked_fill_(hit, -1e30)

    logits = torch.cat([true_logits[:, :, None], sample_logits], -1)

    return logits


if __name__ == '__main__':
    S, B = 3, 4
    n_vocab = 10000
    n_sample = 5
    H = 32

    sampler = LogUniformSampler(n_vocab, unique=True)

    labels = torch.LongTensor(S, B).random_(0, n_vocab)
    embedding = nn.Embedding(n_vocab, H)
    bias = torch.zeros(n_vocab)
    inputs = torch.Tensor(S, B, H).normal_()

    logits, out_labels = sample_logits(embedding, bias, labels, inputs, sampler, n_sample)
    print('logits', logits.detach().numpy().tolist())
    print('logits shape', logits.size())
    print('out_labels', out_labels.detach().numpy().tolist())
    print('out_labels shape', out_labels.size())
