# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Copyright (c) 2018, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0.

"""Projected Adaptive Log-Softmax layer."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class OptionalParameterList(nn.ParameterList):
    def extra_repr(self) -> str:
        child_lines = []

        for k, p in self._parameters.items():
            if p is not None:
                size_str = "x".join(str(size) for size in p.size())
                device_str = "" if not p.is_cuda else " (GPU {})".format(p.get_device())
                parastr = "Parameter containing: [{} of size {}{}]".format(torch.typename(p), size_str, device_str)
                child_lines.append("  (" + str(k) + "): " + parastr)

        tmpstr = "\n".join(child_lines)

        return tmpstr


class ProjectedAdaptiveLogSoftmax(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_embed: int,
        d_model: int,
        cutoffs: Tuple[int],
        tie_projs: Tuple[bool],
        emb_projs: Optional[torch.FloatTensor] = None,
        emb_weights: Optional[torch.FloatTensor] = None,
        div_val: Optional[int] = 1,
        keep_order: Optional[bool] = True,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.d_model = d_model
        self.tie_projs = tie_projs
        self.div_val = div_val
        self.keep_order = keep_order

        self.cutoffs = cutoffs + [vocab_size]
        self.cutoffs_ends = [0] + self.cutoffs

        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.cutoffs[0] + self.n_clusters

        # Whenever clusters are available, we need their weights and biases
        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))

        if not emb_weights:
            self.out_weights = nn.ParameterList()
        else:
            self.out_weights = emb_weights

        self.out_biases = nn.ParameterList()
        self.out_projs = OptionalParameterList()
        self.out_shared_projs = emb_projs

        # Core logic for handling different dividents
        if div_val == 1:
            for i in range(len(self.cutoffs)):
                if d_model != d_embed:
                    if tie_projs[i]:
                        self.out_projs.append(None)
                    else:
                        self.out_projs.append(nn.Parameter(torch.FloatTensor(d_model, d_embed)))
                else:
                    self.out_projs.append(None)

            if not emb_weights:
                self.out_weights.append(nn.Parameter(torch.zeros(vocab_size, d_embed)))

            self.out_biases.append(nn.Parameter(torch.zeros(vocab_size)))
        else:
            for i in range(len(self.cutoffs)):
                cutoff_start, cutoff_end = (
                    self.cutoffs_ends[i],
                    self.cutoffs_ends[i + 1],
                )

                d_embed_i = d_embed // (div_val**i)

                if tie_projs[i]:
                    self.out_projs.append(None)
                else:
                    self.out_projs.append(nn.Parameter(torch.FloatTensor(d_model, d_embed_i)))

                if not emb_weights:
                    self.out_weights.append(nn.Parameter(torch.zeros(cutoff_end - cutoff_start, d_embed_i)))

                self.out_biases.append(nn.Parameter(torch.zeros(cutoff_end - cutoff_start)))

    def _compute_logits(
        self,
        inputs: torch.FloatTensor,
        weight: torch.FloatTensor,
        bias: torch.FloatTensor,
        proj: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if proj is None:
            logits = F.linear(inputs, weight, bias=bias)
        else:
            inputs_proj = F.linear(inputs, proj.t().contiguous())
            logits = F.linear(inputs_proj, weight, bias=bias)

        return logits

    def _get_shared_proj(self, idx: int) -> Union[None, torch.FloatTensor]:
        if self.tie_projs[idx]:
            if len(self.out_shared_projs) == 0:
                return None

            elif len(self.out_shared_projs) == 1:
                return self.out_shared_projs[0]

            else:
                return self.out_shared_projs[idx]

        return self.out_projs[idx]

    def forward(self, inputs: torch.FloatTensor, labels: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        if labels is not None:
            # Shift `n` tokens to predict `n+1`
            inputs = inputs[..., :-1, :].contiguous()
            inputs = inputs.view(-1, inputs.size(-1))

            labels = labels[..., 1:].contiguous()
            labels = labels.view(-1)

            if inputs.size(0) != labels.size(0):
                raise RuntimeError("Inputs and labels should have the same size in the batch dimension.")
        else:
            inputs = inputs.view(-1, inputs.size(-1))

        if self.n_clusters == 0:
            logits = self._compute_logits(
                inputs,
                self.out_weights[0],
                self.out_biases[0],
                self._get_shared_proj(0),
            )

            if labels is not None:
                output = -F.log_softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
            else:
                output = F.log_softmax(logits, dim=-1)
        else:
            # Creates weights and biases to handle all available clusters
            weights, biases = [], []

            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    cutoff_start, cutoff_end = (
                        self.cutoffs_ends[i],
                        self.cutoffs_ends[i + 1],
                    )

                    weight_i = self.out_weights[0][cutoff_start:cutoff_end]
                    bias_i = self.out_biases[0][cutoff_start:cutoff_end]
                else:
                    weight_i = self.out_weights[i]
                    bias_i = self.out_biases[i]

                if i == 0:
                    weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)

            # Defines the head weight, bias and projection
            head_weight, head_bias, head_proj = (
                weights[0],
                biases[0],
                self._get_shared_proj(0),
            )

            # Calculates the head logits and their probabilities
            head_logits = self._compute_logits(inputs, head_weight, head_bias, head_proj)
            head_probs = F.log_softmax(head_logits, dim=1)

            if labels is not None:
                output = torch.zeros_like(labels, dtype=inputs.dtype, device=inputs.device)
            else:
                output = inputs.new_empty((head_logits.size(0), self.vocab_size))

            offset = 0
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                cutoff_start, cutoff_end = cutoff_values[i], cutoff_values[i + 1]

                if labels is not None:
                    # Gathers a mask of valid indexes
                    mask_i = (labels >= cutoff_start) & (labels < cutoff_end)
                    indexes_i = mask_i.nonzero().squeeze()

                    if indexes_i.numel() == 0:
                        continue

                    target_i = labels.index_select(0, indexes_i) - cutoff_start
                    head_probs_i = head_probs.index_select(0, indexes_i)
                    inputs_i = inputs.index_select(0, indexes_i)
                else:
                    inputs_i = inputs

                if i == 0:
                    if labels is not None:
                        probs_i = head_probs_i.gather(1, target_i[:, None]).squeeze(1)
                    else:
                        output[:, : self.cutoffs[0]] = head_probs[:, : self.cutoffs[0]]
                else:
                    weight_i, bias_i, proj_i = (
                        weights[i],
                        biases[i],
                        self._get_shared_proj(i),
                    )

                    tail_logits_i = self._compute_logits(inputs_i, weight_i, bias_i, proj_i)
                    tail_probs_i = F.log_softmax(tail_logits_i, dim=1)

                    cluster_probs_i = self.cutoffs[0] + i - 1

                    if labels is not None:
                        tail_probs_i = tail_probs_i.gather(1, target_i[:, None]).squeeze(1)
                        probs_i = head_probs_i[:, cluster_probs_i] + tail_probs_i
                    else:
                        probs_i = head_probs[:, cluster_probs_i, None] + tail_probs_i

                        output[:, cutoff_start:cutoff_end] = probs_i

                if labels is not None:
                    if self.keep_order:
                        output.index_copy_(0, indexes_i, -probs_i)
                    else:
                        output[offset : offset + probs_i.size(0)].copy_(-probs_i)

                    offset += probs_i.size(0)

        return output
