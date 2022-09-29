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
import torch
import torch.nn as nn
import transformers
import os
import numpy as np
import collections
import yaml
import collections
import argparse
import re
import types
from typing import Optional
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from archai.nlp.models.model_loader import load_model_from_config
from archai.nlp.nas.zero_cost_utils.flops import get_model_flops
from archai.nlp.datasets.lm_iterators import LMOrderedIterator
from archai.nlp.models.mem_transformer.model_mem_transformer import MemTransformerLM
from archai.nlp.models.hf_gpt2.model_hf_gpt2 import HfGPT2Flex


# TODO: handle layerNorm


def get_layer_metric_array(net, metric):
    metric_array = []

    for layer in net.modules():
        if isinstance(layer, transformers.Conv1D) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))

    return metric_array


def compute_synflow_per_weight(net, inputs, targets):
    device = inputs.device

    # convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    # convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if "weight_mask" not in name:
                param.mul_(signs[name])

    # keep signs of all params
    signs = linearize(net)

    # Compute gradients with input of 1s
    net.zero_grad()
    net.double()

    outputs = net(inputs, targets, mems=None)

    # TODO: should this be loss or logits?
    if isinstance(net, MemTransformerLM):
        torch.sum(outputs).backward()
    elif isinstance(net, HfGPT2Flex):
        torch.sum(outputs.logits).backward()
    else:
        raise NotImplementedError

    # select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
            # return torch.abs(layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, synflow)

    # apply signs of all params
    nonlinearize(net, signs)

    return grads_abs


def forward_crit(self, hidden, target=None, keep_order=False, output_loss=True, output_prediction_scores=False):
  '''
      hidden :: [len*bsz x d_proj]
  '''
  if self.n_clusters == 0:
    logit = self._compute_logit(hidden, self.out_layers_weights[0], self.out_layers_biases[0], self.get_out_proj(0))
    return None, logit
  else:
    # construct weights and biases
    weights, biases = [], []
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

    head_weight, head_bias, head_proj = weights[0], biases[0], self.get_out_proj(0)
    head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
    return None, head_logit


def forward_synflow_gpt(self, data, target, mems, output_loss=True, output_prediction_scores=False):
    # Causal attention mask is created inside the model
    emb_ = self.model.transformer.wte(data)
    word_emb = torch.ones_like(emb_)
   
    outputs = self.model(
        inputs_embeds=word_emb,
        labels=target,
        # attention_mask=torch.ones_like(data),
        output_loss=output_loss,
        output_prediction_scores=output_prediction_scores,
    )

    return outputs


def _forward_synflow_memformer(self, dec_inp, mems=None, past_key_values=None):
    qlen, bsz = dec_inp.size()

    word_emb = self.word_emb(dec_inp)

    mlen = mems[0].size(0) if mems is not None else 0
    plen = past_key_values[0][0].size(0) if past_key_values[0] is not None else 0
    klen = mlen + qlen
    # `plen` should be taken into account when creating the
    # attention mask because `past_key_values` might be used
    if self.same_length:
        all_ones = word_emb.new_ones(qlen, klen+plen)
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

        # mask everything to all one for synflow
        core_out = torch.ones_like(core_out, dtype=core_out.dtype, device=core_out.device)
        pos_emb = torch.ones_like(pos_emb, dtype=pos_emb.dtype, device=pos_emb.device)

        for i, (layer, past_key_values_i) in enumerate(zip(self.layers, past_key_values)):
            hids.append(core_out.detach())
            mems_i = None if mems is None else mems[i]
            core_out, past_key_values_i = layer(core_out, pos_emb, getattr(self, f'r_w_bias_{i}'),
                                                getattr(self, f'r_r_bias_{i}'), dec_attn_mask=dec_attn_mask,
                                                mems=mems_i, past_key_values=past_key_values_i)
            pasts_key_values = pasts_key_values + (past_key_values_i, )
    else:
        raise NotImplemented

    core_out = self.drop(core_out)

    new_mems = self._update_mems(hids, mems, qlen, mlen)

    return core_out, new_mems, pasts_key_values


def forward_synflow_memformer(self, input_ids:torch.Tensor, labels:Optional[torch.Tensor]=None, mems:Optional[torch.Tensor]=None,
                past_key_values:Optional[torch.Tensor]=None, output_loss=True, output_prediction_scores=False):
    # input_ids and labels are transposed within the code to avoid major changes
    # input_ids -> [seq_len, batch_size], labels -> [seq_len, batch_size]
    # Returns:
    # loss -> [batch_size, seq_len], prediction_scores -> [batch_size, seq_len, vocab_size]
    # nn.DataParallel does not allow size(0) tensors to be broadcasted.
    # So, have to initialize size(0) mems inside the model forward.
    # Moreover, have to return new_mems to allow nn.DataParallel to piece
    # them together.

    # Transposes `input_ids` and `labels` to seq_len x batch_size
    input_ids = input_ids.t()
    if labels is not None:
        labels = labels.t()

    if mems is None:
        mems = self.init_mems()

    if labels is None:
        output_loss = False

    if past_key_values is None:
        past_key_values = tuple([None] * self.n_layer)

    hidden, mems, past_key_values = self._forward(input_ids, mems=mems, past_key_values=past_key_values)

    tgt_len = labels.size(0) if labels is not None else input_ids.size(0)
    pred_hid = hidden[-tgt_len:]
    _, out = self.crit(pred_hid.view(-1, pred_hid.size(-1)))
    out = out.view(tgt_len, -1)
    
    return out


def get_synflow_scores(args, exp_name):
    path_to_results = exp_name
    yaml_file_scores = os.path.join(path_to_results, "synflow_scores_seed_{}.yaml".format(args.seed))
    yaml_file_cost = os.path.join(path_to_results, "synflow_cost.yaml")
    calc_scores = not os.path.exists(yaml_file_scores)
    calc_costs = args.get_cost and not os.path.exists(yaml_file_cost)

    device = torch.device("cpu")

    files = []
    dirlist = [path_to_results]
    while len(dirlist) > 0:
        for (dirpath, dirnames, filenames) in os.walk(dirlist.pop()):
            dirlist.extend([os.path.join(dirpath, d) for d in dirnames])
            files.extend(
                map(lambda n: os.path.join(*n), zip([dirpath] * len(filenames), filenames),))

    if calc_scores or calc_costs:
        scores = {}
        costs = {}
        count = 1
        for _f in set(files):
            if "model_config.yaml" in _f:
                idx =  re.search('(config_[0-9]+)', _f).span()[0]
                job = _f[idx:]
                config_name = job.split('/')[0]
                config_name += '_' + job.split('/')[1]

                with open(_f, "r") as f:
                    model_config = yaml.full_load(f)
                model = load_model_from_config(args.model_type, model_config)
                model.n_token = model_config["n_token"]

                if isinstance(model, MemTransformerLM):
                    model._forward = types.MethodType(_forward_synflow_memformer, model)
                    model.forward = types.MethodType(forward_synflow_memformer, model)
                    model.crit.forward = types.MethodType(forward_crit, model.crit)

                elif isinstance(net, HfGPT2Flex):
                    model.forward = types.MethodType(forward_synflow_gpt, model)
                    model.model.lm_head.forward = types.MethodType(forward_crit, model.model.lm_head)

                B = 1
                tgt_len, mem_len, ext_len = (model_config["tgt_len"], model_config["mem_len"], model_config["ext_len"],)
                data_len = tgt_len
                data = torch.ones(data_len * B).to(device, torch.long)
                diter = LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)
                if calc_scores:
                    for idx, (inp, tgt, seqlen, _) in enumerate(diter):
                        grads_abs = compute_synflow_per_weight(model, inp, tgt)
                        score = np.sum([torch.sum(g).detach().numpy() for g in grads_abs])
                        break
                    scores[config_name] = score.tolist()
                if calc_costs:
                    model.eval()
                    with torch.no_grad():
                        for _, (inp, tgt, _, _) in enumerate(diter):
                            curr_flops = get_model_flops(model, inp, tgt)
                            total_flops = np.sum([curr_flops[k] for k in ["Attn", "FFN", "Sftmax"]]).tolist()
                            break
                    costs[config_name] = 3 * total_flops
                    print(count, config_name, 'score:', scores[config_name], 'FLOPS:', costs[config_name])
                else:
                    print(count, config_name, 'score:', scores[config_name])
                count += 1

    if calc_scores:
        with open(yaml_file_scores, "w") as f:
            yaml.dump(scores, f)
    if calc_costs:
        with open(yaml_file_cost, "w") as f:
            yaml.dump(costs, f)