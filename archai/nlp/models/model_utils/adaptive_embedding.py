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

"""Adaptive Embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                 sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = self.clean_cutoffs(cutoffs, n_token)
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=(sample_softmax > 0))
            )
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed).zero_()))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx-l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i).zero_()))

    def default_cutoffs(self, n_token):
        return [19997, 39997, 199997, n_token]

    def clean_cutoffs(self, cutoffs, n_token):
        if cutoffs is None:
            cutoffs = self.default_cutoffs(n_token)

        cutoffs = cutoffs.copy()
        if not cutoffs:
            cutoffs = [n_token]
        assert isinstance(cutoffs, list) and len(cutoffs) > 0

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

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            # Makes sure that `inp_flat` is spanned across a contiguous dimension
            # due to the possiiblity of having different layer sizes 
            inp_flat = inp.contiguous().view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj],
                                   dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero(as_tuple=False).squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i]).to(emb_flat.dtype)

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed
