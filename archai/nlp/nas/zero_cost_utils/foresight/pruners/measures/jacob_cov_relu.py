# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from builtins import isinstance
import copy
import types
from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from archai.nlp.models.mem_transformer.model_mem_transformer import MemTransformerLM
# from archai.nlp.nvidia_transformer_xl.zero_cost_nas.synflow import forward_crit

from . import measure

class OneHotLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(OneHotLinear, self).__init__(in_features, out_features, bias, device, dtype)

    def forward(self, input):
        self.input_oh = torch.index_select(torch.eye(self.in_features), 0, input.view(-1).long()).view(input.size()+(self.in_features,))
        self.input_oh.requires_grad = True
        return F.linear(self.input_oh, self.weight, self.bias)


def forward_jacob_cov(self,
                input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                mems: Optional[torch.Tensor] = None,
                past_key_values: Optional[torch.Tensor] = None,
                output_loss: Optional[bool] = False,
                output_prediction_scores: Optional[bool] = True
                ) -> Tuple[torch.Tensor, ...]:
        assert mems is None, 'HfGPT2Flex does not support memory (mems).'

        labels = None

        # Causal attention mask is created inside the model
        outputs = self.model(input_ids=input_ids,
                             labels=labels,
                             attention_mask=torch.ones_like(input_ids),
                             past_key_values=past_key_values,
                             output_loss=output_loss,
                             output_prediction_scores=output_prediction_scores)

        return self.fc_to_1class(outputs.logits)
        


def forward_crit(self, hidden, target=None, keep_order=False):
  '''
      hidden :: [len*bsz x d_proj]
  '''
  if self.n_clusters == 0:
    logit = self._compute_logit(hidden, self.out_layers_weights[0], self.out_layers_biases[0], self.get_out_proj(0))
    return logit
  else:
    # construct weights and biases
    weights, biases, projs = [], [], []
    for i in range(len(self.cutoffs)):
        if self.div_val == 1:
            l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
            weight_i = self.out_layers_weights[0][l_idx:r_idx]
            bias_i = self.out_layers_biases[0][l_idx:r_idx]
        else:
            weight_i = self.out_layers_weights[i]
            bias_i = self.out_layers_biases[i]

        if i == 0:
            weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
            bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)

        weights.append(weight_i)
        biases.append(bias_i)
        projs.append(self.get_out_proj(i))

    head_weight, head_bias, head_proj = weights[0], biases[0], projs[0]
    head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)

    out = hidden.new_empty((head_logit.size(0), self.n_token))
    out[:, :self.shortlist_size] = head_logit[:, :self.shortlist_size]

    for i, (start_idx, stop_idx) in enumerate(zip(self.cutoffs, self.cutoffs[1:])):
        weight_i, bias_i, proj_i = weights[i+1], biases[i+1], projs[i+1]
        cluster_output = self._compute_logit(hidden, weight_i, bias_i, proj_i) #self.tail[i](input)
        out[:, start_idx:stop_idx] = cluster_output

    return out

def modify_net(net, jacob_size):
    # for idx, l in enumerate(net.word_emb.emb_layers):
    #     if isinstance(l, nn.Embedding):
    #         print('found one')
    #         assert l.sparse==False, 'sparse embedding cannot be converted to nn.Linear layer'
    #         new_layer = OneHotLinear(l.num_embeddings, l.embedding_dim, bias=False)#EmbeddingMul(l.num_embeddings, l.embedding_dim, _weight=l.weight.data)
    #         new_layer.weight.data = copy.deepcopy(l.weight.data.transpose(1, 0))
    #         # new_layer.to(l.device)
    #         net.word_emb.emb_layers[idx] = new_layer
    #         del l
    
    net.forward = types.MethodType(forward_jacob_cov, net)
    # net.crit.forward = types.MethodType(forward_crit, net.crit)
    net.fc_to_1class = torch.nn.Linear(net.n_token, 1, bias=False)
    
    net.K = np.zeros((jacob_size, jacob_size))
    net.compute = 0

    def counting_forward_hook(module, inp, out):
        try:
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0)*inp.size(1), -1)
            x = (inp > 0).float()
            K = x @ x.t()
            K2 = (1.-x) @ (1.-x.t())
            net.K = net.K + K.cpu().numpy() + K2.cpu().numpy()
            net.compute += 2 * (jacob_size * jacob_size * inp.size(-1)) + (jacob_size ** 3)
        except:
            pass

    for module in net.modules():
        if 'NewGELUActivation' in str(type(module)):
            module.register_forward_hook(counting_forward_hook)

def get_batch_jacobian(net, x, target, device, split_data):
    net.zero_grad()

    x_emb = net.word_emb(x)
    word_emb = x_emb.data
    word_emb.requires_grad_(True)

    N = x.shape[-1]
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data
        y, _ = net(word_emb[:, st:en], target[:, st:en], mems=None)
        # y, _ = net(word_emb[:, st:en, :], target[:, st:], mems=None)
        y.backward(torch.ones_like(y))

    jacob = word_emb.grad.detach()
    word_emb.requires_grad_(False)
    return jacob, target.detach()

def eval_score(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    return ld

@measure('jacob_cov_relu', bn=True, copy_net=False)
def compute_jacob_cov_relu(net, inputs, targets, split_data=1, loss_fn=None, maxofn=1):
    device = inputs.device
    # Compute gradients (but don't apply them)
    modify_net(net, jacob_size=inputs.size(0)*inputs.size(1))
    net.zero_grad()
    net.to(device)

    # jacobs, labels = get_batch_jacobian(net, inputs, targets, device, split_data=split_data)
    # jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
    # try:
    #     jc = float(eval_score(jacobs, labels))
    # except Exception as e:
    #     print(e)
    #     jc = float(np.nan)
    
    s = []
    for j in range(maxofn):
        net.compute = 0
        net(inputs.to(device), targets.to(device))
        s.append(eval_score(net.K, targets.detach()))
    
    jc = float(np.mean(s))
    return jc
