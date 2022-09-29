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

import types
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from archai.nlp.models.mem_transformer.model_mem_transformer import MemTransformerLM
from archai.nlp.models.hf_gpt2.model_hf_gpt2 import HfGPT2Flex

from . import measure


class OneHotLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(OneHotLinear, self).__init__(
            in_features, out_features, bias, device, dtype
        )

    def forward(self, input):
        self.input_oh = torch.index_select(
            torch.eye(self.in_features), 0, input.view(-1).long()
        ).view(input.size() + (self.in_features,))
        self.input_oh.requires_grad = True
        return F.linear(self.input_oh, self.weight, self.bias)


def forward_after_embedding_memformer(self, word_emb, labels, mems=None, past_key_values=None):
    if labels is not None:
        labels = labels.t()

    if mems is None:
        mems = self.init_mems()
    if past_key_values is None:
            past_key_values = tuple([None] * self.n_layer)

    qlen, bsz = word_emb.size(0), word_emb.size(1)

    mlen = mems[0].size(0) if mems is not None else 0
    plen = past_key_values[0][0].size(0) if past_key_values[0] is not None else 0
    klen = mlen + qlen
    if self.same_length:
        all_ones = word_emb.new_ones(qlen, klen)
        mask_len = klen - self.mem_len - 1
        if mask_len > 0:
            mask_shift_len = qlen - mask_len
        else:
            mask_shift_len = qlen
        dec_attn_mask = (torch.triu(all_ones, 1+mlen+plen)
                             + torch.tril(all_ones, -mask_shift_len)).bool()
    else:
        dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen+plen), diagonal=1+mlen+plen).bool()

    hids = []
    pasts_key_values = ()
    # default
    if self.attn_type == 0:
        pos_seq = torch.arange(klen+plen-1, plen-1, -1.0, device=word_emb.device,
                                dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        core_out = self.drop(word_emb)
        pos_emb = self.drop(pos_emb)

        for i, (layer, past_key_values_i) in enumerate(zip(self.layers, past_key_values)):
            hids.append(core_out.detach())
            mems_i = None if mems is None else mems[i]
            core_out, past_key_values_i = layer(core_out, pos_emb, getattr(self, f'r_w_bias_{i}'),
                                                getattr(self, f'r_r_bias_{i}'), dec_attn_mask=dec_attn_mask,
                                                mems=mems_i, past_key_values=past_key_values_i)
            pasts_key_values = pasts_key_values + (past_key_values_i, )
    # learnable
    elif self.attn_type == 1:
        core_out = self.drop(word_emb)
        for i, (layer, past_key_values_i) in enumerate(zip(self.layers, past_key_values)):
            hids.append(core_out.detach())
            if self.clamp_len > 0:
                r_emb = getattr(self, f'r_emb_{i}')[-self.clamp_len:]
                r_bias = getattr(self, f'r_bias_{i}')[-self.clamp_len:]
            else:
                r_emb, r_bias = getattr(self, f'r_emb_{i}'), getattr(self, f'r_bias_{i}')

            mems_i = None if mems is None else mems[i]
            core_out, past_key_values_i = layer(core_out, r_emb, getattr(self, f'r_w_bias_{i}'),
                                                r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i,
                                                past_key_values=past_key_values_i)
            pasts_key_values = pasts_key_values + (past_key_values_i, )
    # absolute
    elif self.attn_type == 2:
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        core_out = self.drop(word_emb + pos_emb[-qlen:])

        for i, (layer, past_key_values_i) in enumerate(zip(self.layers, past_key_values)):
            hids.append(core_out.detach())
            mems_i = None if mems is None else mems[i]
            if mems_i is not None and len(mems_i) and i == 0:
                mems_i += pos_emb[:mlen]
            core_out, past_key_values_i = layer(core_out, dec_attn_mask=dec_attn_mask,
                                                mems=mems_i, past_key_values=past_key_values_i)
            pasts_key_values = pasts_key_values + (past_key_values_i, )
    elif self.attn_type == 3:
        core_out = self.drop(word_emb)

        for i, (layer, past_key_values_i) in enumerate(zip(self.layers, past_key_values)):
            hids.append(core_out.detach())
            mems_i = None if mems is None else mems[i]
            if mems_i is not None and len(mems_i) and mlen > 0:
                cur_emb = self.r_emb[i][:-qlen]
                cur_size = cur_emb.size(0)
                if cur_size < mlen:
                    cur_emb_pad = cur_emb[0:1].expand(mlen-cur_size, -1, -1)
                    cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                else:
                    cur_emb = cur_emb[-mlen:]
                mems_i += cur_emb.view(mlen, 1, -1)
            core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

            core_out, past_key_values_i = layer(core_out, dec_attn_mask=dec_attn_mask,
                                                mems=mems_i, past_key_values=past_key_values_i)
            pasts_key_values = pasts_key_values + (past_key_values_i, )

    core_out = self.drop(core_out)
    # core_out = self.fc_to_1class(core_out)

    new_mems = self._update_mems(hids, mems, qlen, mlen)

    tgt_len = labels.size(0)
    pred_hid = core_out[-tgt_len:]
    _, out = self.crit(pred_hid.view(-1, pred_hid.size(-1)))
    # out = out.view(tgt_len, -1)
    out = self.fc_to_1class(out)

    return out, None


def forward_after_embedding_gpt2(self, word_emb, target, mems=None):
    # Causal attention mask is created inside the model
    outputs = self.model(
        labels=target,
        inputs_embeds=word_emb,
        output_loss=False,
        output_prediction_scores=True,
    )

    return self.fc_to_1class(outputs.logits), None


def forward_crit(self, hidden, target=None, keep_order=False, output_loss=True, output_prediction_scores=False):
    """
    hidden :: [len*bsz x d_proj]
    """
    if self.n_clusters == 0:
        logit = self._compute_logit(
            hidden,
            self.out_layers_weights[0],
            self.out_layers_biases[0],
            self.get_out_proj(0),
        )
        return None, logit
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
        out[:, : self.shortlist_size] = head_logit[:, : self.shortlist_size]

        for i, (start_idx, stop_idx) in enumerate(zip(self.cutoffs, self.cutoffs[1:])):
            weight_i, bias_i, proj_i = weights[i + 1], biases[i + 1], projs[i + 1]
            cluster_output = self._compute_logit(
                hidden, weight_i, bias_i, proj_i
            )  # self.tail[i](input)
            out[:, start_idx:stop_idx] = cluster_output

        return None, out


def modify_net(net):
    # for idx, l in enumerate(net.word_emb.emb_layers):
    #     if isinstance(l, nn.Embedding):
    #         assert l.sparse==False, 'sparse embedding cannot be converted to nn.Linear layer'
    #         new_layer = OneHotLinear(l.num_embeddings, l.embedding_dim, bias=False)#EmbeddingMul(l.num_embeddings, l.embedding_dim, _weight=l.weight.data)
    #         new_layer.weight.data = copy.deepcopy(l.weight.data.transpose(1, 0))
    #         # new_layer.to(l.device)
    #         net.word_emb.emb_layers[idx] = new_layer
    #         del l

    if isinstance(net, MemTransformerLM):
        net.forward = types.MethodType(forward_after_embedding_memformer, net)
        net.crit.forward = types.MethodType(forward_crit, net.crit)
    elif isinstance(net, HfGPT2Flex):
        net.forward = types.MethodType(forward_after_embedding_gpt2, net)
        net.model.lm_head.forward = types.MethodType(forward_crit, net.model.lm_head)
    else:
        raise NotImplementedError
    net.fc_to_1class = torch.nn.Linear(net.n_token, 1, bias=False)
    return net


def get_batch_jacobian(net, x, target, device, split_data):
    net.zero_grad()

    if isinstance(net, MemTransformerLM):
        x = x.t()
        x_emb = net.word_emb(x)
    if isinstance(net, HfGPT2Flex):    
        x_emb = net.model.transformer.wte(x)
    word_emb = x_emb.data
    word_emb.requires_grad_(True)

    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y, _ = net(word_emb[st:en, :], target[st:en, :], mems=None)
        # y, _ = net(word_emb[:, st:en, :], target[:, st:], mems=None)
        y.backward(torch.ones_like(y))

    jacob = word_emb.grad.detach()
    word_emb.requires_grad_(False)
    return jacob, target.detach()


def eval_score(jacob, labels=None):
    corrs = np.corrcoef(jacob)
    v, _ = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1.0 / (v + k))


@measure("jacob_cov", bn=True)
def compute_jacob_cov(net, inputs, targets, split_data=1, loss_fn=None):
    device = inputs.device
    # Compute gradients (but don't apply them)
    net = modify_net(net).to(device)
    net.zero_grad()

    jacobs, labels = get_batch_jacobian(
        net, inputs, targets, device, split_data=split_data
    )
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

    try:
        jc = float(eval_score(jacobs, labels))
    except Exception as e:
        print(e)
        jc = float(np.nan)

    return jc
