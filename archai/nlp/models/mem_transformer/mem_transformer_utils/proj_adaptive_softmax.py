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

"""Projected Adaptive Log-Softmax layer.
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class OptionalParameterList(nn.ParameterList):
    """Overrides PyTorch ParameterList to avoid printing `None` parameters.

    """

    def extra_repr(self) -> str:
        """Avoids printing `None` parameters.

        Returns:
            String representation of the class.
            
        """

        child_lines = []

        for k, p in self._parameters.items():
            if p is not None:
                size_str = 'x'.join(str(size) for size in p.size())
                device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.get_device())
                parastr = 'Parameter containing: [{} of size {}{}]'.format(
                    torch.typename(p), size_str, device_str)
                child_lines.append('  (' + str(k) + '): ' + parastr)

        tmpstr = '\n'.join(child_lines)

        return tmpstr


class ProjectedAdaptiveLogSoftmax(nn.Module):
    """Implements a Projected Adaptive Log-Softmax layer.

    """

    def __init__(self,
                 n_token: int,
                 d_embed: int,
                 d_proj: int,
                 cutoffs: Optional[List[int]],
                 adaptive: bool,
                 div_val: Optional[int] = 1,
                 tie_projs: Optional[List[bool]] = None,
                 out_layers_weights: Optional[torch.Tensor] = None,
                 out_projs: Optional[torch.Tensor] = None,
                 keep_order: Optional[bool] = False) -> None:
        """Overrides method with a custom implementation.

        Args:
            n_token: Number of tokens (vocabulary size).
            d_embed: Dimensionality of the embeddings.
            d_proj: Dimensionality of the inputs states.
            cutoffs: Cutoffs for the adaptive softmax.
            adaptive: Whether to use adaptive softmax or not.
            div_val: Divident value for adapative input and softmax.
            tie_projs: Whether to tie projections or not.
            out_layers_weights: Tensor holding the embedding layer weights.
            out_projs: Tensor holding the embedding layer projections.
            keep_order: Whether to keep the order of the outputs.

        """

        super().__init__()

        self.n_token = n_token # vocab size
        self.d_embed = d_embed
        self.d_proj = d_proj

        self.cutoffs = ProjectedAdaptiveLogSoftmax.clean_cutoffs(cutoffs, n_token)

        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val

        self.shortlist_size = self.cutoffs[0] # TODO: remove? Never used...
        self.n_clusters = len(self.cutoffs) - 1 # number of clusters will be >= 0
        self.head_size = self.shortlist_size + self.n_clusters # TODO: remove? Never used...

        self.tie_projs = ProjectedAdaptiveLogSoftmax.clean_tie_projs(tie_projs,
            self.cutoffs, adaptive, n_token)
        assert len(self.tie_projs) == len(self.cutoffs)

        if self.n_clusters > 0: # create parameters for each cluster
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))

        if not out_layers_weights: # if not shared weights with embedding layer
            self.out_layers_weights = nn.ParameterList()
        else:
            self.out_layers_weights = out_layers_weights

        self.out_layers_biases = nn.ParameterList()

        self.shared_out_projs = out_projs # default is empty list
        self.out_projs = OptionalParameterList()

        if div_val == 1: # if cluster sizes are all same
            if d_proj != d_embed: # default: d_proj is same as d_model
                for i in range(len(self.cutoffs)):
                    if tie_projs[i]:
                        self.out_projs.append(None)
                    else:
                        self.out_projs.append(
                            nn.Parameter(torch.zeros(d_proj, d_embed))
                        )
            else: # no projection required
                # self.out_projs = [None] * len(self.cutoffs)
                self.out_projs.append(None)

            self.out_layers_biases.append(
                nn.Parameter(torch.zeros(n_token))
                )

            if not out_layers_weights:
                self.out_layers_weights.append(
                    nn.Parameter(torch.zeros(n_token, d_embed))
                    )
        else:
            for i in range(len(self.cutoffs)):
                # calculate embeddings for clusters, exponentially decreasing in size
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)

                if tie_projs[i]: # True for non-head clusters in Wiki* datasets but False for lm1b
                    self.out_projs.append(None)
                else:
                    self.out_projs.append(
                        nn.Parameter(torch.zeros(d_proj, d_emb_i))
                    )

                self.out_layers_biases.append(
                    nn.Parameter(torch.zeros(r_idx - l_idx))
                    )
                if not out_layers_weights:
                    self.out_layers_weights.append(
                        nn.Parameter(torch.zeros(r_idx - l_idx, d_emb_i))
                        )

        self.keep_order = keep_order

    def _compute_logit(self,
                       hidden: torch.Tensor,
                       weight: torch.Tensor,
                       bias: torch.Tensor,
                       proj: torch.Tensor) -> torch.Tensor:
        """Computes the logits through a linear activation.

        Args:
            hidden: Input tensor with hidden states.
            weight: Weight tensor.
            bias: Bias tensor.
            proj: Projection tensor.

        Returns:
            (torch.Tensor): Output logits.

        """

        # if no projection then simply multiply hidden values with wights
        # else apply projection to hidden and then multiply with weight matrix
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)
        else:
            # below is equivalent to:
            # proj_hid = nn.functional.linear(hidden, proj.t().contiguous())
            # logit = nn.functional.linear(proj_hid, weight, bias=bias)
            logit = torch.einsum('bd,de,ev->bv', (hidden, proj, weight.t()))
            if bias is not None:
                logit = logit + bias
        return logit

    @staticmethod
    def default_cutoffs(n_token: int) -> List[int]:
        """Gathers the default cutoffs.

        Args:
            n_token: Maximum number of tokens.

        Returns:
            (List[int]): Cutoffs.

        """

        return [19997, 39997, 199997, n_token]

    @staticmethod
    def default_tie_proj(cutoffs: List[int],
                         adaptive: bool) -> List[bool]:
        """Gathers the default tied projections.

        Args:
            cutoffs: Cutoffs.
            adaptive: Whether to use adaptive softmax or not.

        Returns:
            (List[bool]): Default tied projections.

        """

        return [not adaptive] + [True] * (len(cutoffs)-1)

    @staticmethod
    def clean_cutoffs(cutoffs: Optional[List[int]],
                      n_token: int) -> List[int]:
        """Cleans the cutoffs.

        Args:
            cutoffs: Cutoffs.
            n_token: Maximum number of tokens.

        Returns:
            (List[int]): Cleaned cutoffs.
            
        """

        if cutoffs is None:
            cutoffs = ProjectedAdaptiveLogSoftmax.default_cutoffs(n_token)

        cutoffs = cutoffs.copy()
        if not cutoffs:
            cutoffs = [n_token]
        assert isinstance(cutoffs, list) and len(cutoffs)>0

        # check if all entries in array are monotonically increasing
        # if any entry is > n_token then we trim the array at that point
        last_co, c = cutoffs[0], 1
        while c < len(cutoffs):
            assert cutoffs[c] > last_co, f"cutoff at {c} is <= {c-1}"
            last_co = cutoffs[c]
            if cutoffs[c] > n_token:
                break
            c += 1
        cutoffs = cutoffs[:c] # trim the list if there was any entry > n_token
        # make sure the last entry is n_token
        if cutoffs[-1] > n_token:
            cutoffs[-1] = n_token
        if cutoffs[-1] < n_token:
            cutoffs.append(n_token)

        return cutoffs

    @staticmethod
    def clean_tie_projs(tie_projs: Optional[List[bool]],
                        cutoffs: List[int],
                        adaptive: bool,
                        n_token: int) -> List[bool]:
        """Cleans the tied projections.

        Args:
            tie_projs: Tied projections.
            cutoffs: Cutoffs.
            adaptive: Whether to use adaptive softmax or not.
            n_token: Maximum number of tokens.

        Returns:
            (List[bool]): Cleaned tied projections.

        """

        if not tie_projs:
            tie_projs = ProjectedAdaptiveLogSoftmax.default_tie_proj(cutoffs, adaptive)
        assert isinstance(tie_projs, list)

        return tie_projs[:len(cutoffs)]

    def get_out_proj(self, i: int) -> Union[None, torch.FloatTensor]:
        """Gathers a shared projection whenever it is valid and available.

        Args:
            i: Index of the projection.

        Returns:
            (Union[None, torch.FloatTensor]): Tensor holding the projection.

        """

        if self.tie_projs[i]:
            if len(self.shared_out_projs) == 0:
                return None
            elif len(self.shared_out_projs) == 1:
                return self.shared_out_projs[0]
            else:
                return self.shared_out_projs[i]
        else:
            return self.out_projs[i]

    def forward(self,
                hidden: torch.Tensor,
                target: torch.Tensor,
                keep_order: Optional[bool] = False,
                output_loss: Optional[bool] = True,
                output_prediction_scores: Optional[bool] = False) -> torch.Tensor:
        """Forward pass through the class.

        Args:
            hidden: Input tensor with hidden states of shape [len*bsz, d_proj]
            target: Labels tensor with shape of [len*bsz]
            keep_order: Whether to keep the order of the outputs.
            output_loss: Whether to output the loss.
            output_prediction_scores: Whethet to output the prediction scores (logits).

        Returns:
            (torch.Tensor): Output tensor.

        """

        if target is not None and hidden.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')
        if target is None:
            output_loss = False

        if self.n_clusters == 0: # if no adaptive softmax
            # compute logits and log_probs as usual
            logit = self._compute_logit(hidden, self.out_layers_weights[0],
                                        self.out_layers_biases[0], self.get_out_proj(0))

            # nll will be None if target is None
            nll = -F.log_softmax(logit, dim=-1) \
                        .gather(1, target.unsqueeze(1)).squeeze(1) \
                            if output_loss else None

            log_probs = F.log_softmax(logit, dim=-1) \
                if output_prediction_scores else None
        else:
            # build list of output weights and biases to use for each cluster
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers_weights[0][l_idx:r_idx]
                    bias_i = self.out_layers_biases[0][l_idx:r_idx]
                else:
                    weight_i = self.out_layers_weights[i]
                    bias_i = self.out_layers_biases[i]

                # for the first cluster we append n_cluster-1 weights and biases for the following clusters
                if i == 0: # wight_i->[cutoff, d_embd], cluster_weight->[n_cluster, d_embed]
                    weight_i = torch.cat(
                        [weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat(
                        [bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)

            # Calculate logits and log probabilities for the head cluster
            head_weight, head_bias, head_proj = weights[0], biases[0], self.get_out_proj(0)
            # end of these logits and log prob is for each of the n_cluster-1
            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj) #[seq_len*batch, cluster(0) size + n_cluster-1]
            head_logprob = F.log_softmax(head_logit, dim=1)

            # prepare the array to fill for nll and log_prob
            # nll is None is target is None
            nll = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device) \
                    if output_loss else None
            log_probs = hidden.new_empty((head_logit.size(0), self.n_token)) \
                    if output_prediction_scores else None

            offset = 0
            cutoff_values = [0] + self.cutoffs # append 0 index for start of first cluster
            for i in range(len(cutoff_values) - 1):
                # calculate range of current cluster
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

                if target is not None:
                    # select the indices in the target where we have labels present in current cluster
                    mask_i = (target >= l_idx) & (target < r_idx)
                    indices_i = mask_i.nonzero(as_tuple=False).squeeze()

                    # if target has no labels from current cluster, no loss or log prob to compute
                    if indices_i.numel() == 0:
                        continue

                    # select parts of relevant target and log prob tensors
                    target_i = target.index_select(0, indices_i) - l_idx
                    head_logprob_i = head_logprob.index_select(0, indices_i)
                    hidden_i = hidden.index_select(0, indices_i)

                # now compute nll_i and fill up log_probs
                # nll_i will be used to fill up nll tensor
                if i == 0:
                    # for the cluster 0, we already have computed log prob, so just
                    # pick out the values relevant to target and stuff into final answer
                    if output_loss:
                        nll_i = -head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                    else:
                        nll_i = None

                    if output_prediction_scores:
                        log_probs[:, : self.cutoffs[0]] = head_logprob[:, : self.cutoffs[0]]
                    # else no log_probs
                else:
                    # select the index in head_logprob where we will put cluster probabilities
                    # original code has bug for using -i instead of cluster_prob_idx
                    cluster_prob_idx = self.cutoffs[0] + i - 1  # No probability for the head cluster

                    weight_i, bias_i, proj_i = weights[i], biases[i], self.get_out_proj(i)

                    if output_loss:
                        tail_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                        tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)

                        nll_i = - (head_logprob_i[:, cluster_prob_idx] \
                                + tail_logprob_i.gather(1, target_i[:, None]).squeeze(1))
                    else:
                        nll_i = None

                    if output_prediction_scores:
                        # for computing log prob, we need to use full hidden layer
                        tail_logit_f = self._compute_logit(hidden, weight_i, bias_i, proj_i)
                        tail_logprob_f = F.log_softmax(tail_logit_f, dim=1)

                        log_probs_i = head_logprob[:, cluster_prob_idx, None] + tail_logprob_f
                        log_probs[:, l_idx:r_idx] = log_probs_i
                    else:
                        log_probs = None

                if output_loss:
                    if self.keep_order or keep_order:
                        nll.index_copy_(0, indices_i, nll_i)
                    else:
                        nll[offset:offset+nll_i.size(0)].copy_(nll_i)
                    offset += nll_i.size(0)

        return (nll, log_probs)
