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


def get_some_data(args, train_dataloader, num_batches, device):
    traindata = []
    if args.dataset != "lm1b":
        train_iter = train_dataloader.get_fixlen_iter(start=0)
    else:
        train_iter = train_dataloader

    for batch, (data, target, seq_len, _) in enumerate(train_iter, start=1):
        if batch > num_batches:
            break
        traindata.append((data, target))
    inputs = torch.cat([a for a, _ in traindata], dim=-1)
    targets = torch.cat([b for _, b in traindata], dim=-1)
    inputs = inputs.to(device)
    targets = targets.to(device)
    return inputs, targets


def get_some_data_grasp(args, train_dataloader, num_classes, samples_per_class, device):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    if args.dataset != "lm1b":
        if args.varlen:
            train_iter = train_dataloader.get_varlen_iter(start=0)
        else:
            train_iter = train_dataloader.get_fixlen_iter(start=0)
    else:
        train_iter = train_dataloader

    while True:
        inputs, targets, _, _ = next(train_iter)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        print(inputs.size())
        for idx in range(inputs.shape[0]):
            x, y = inputs[:, idx : idx + 1], targets[:, idx : idx + 1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    x = torch.cat([torch.cat(_, 0) for _ in datas]).to(device)
    y = torch.cat([torch.cat(_) for _ in labels]).view(-1).to(device)
    return x, y


def get_layer_metric_array(net, metric, mode, include_embedding=False):
    metric_array = []

    for layer in net.modules():
        if mode == "channel" and hasattr(layer, "dont_ch_prune"):
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, transformers.Conv1D) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))
        if isinstance(layer, nn.Embedding) and include_embedding:
            metric_array.append(metric(layer))

    return metric_array


def reshape_elements(elements, shapes, device):
    def broadcast_val(elements, shapes):
        ret_grads = []
        for e, sh in zip(elements, shapes):
            ret_grads.append(
                torch.stack([torch.Tensor(sh).fill_(v) for v in e], dim=0).to(device)
            )
        return ret_grads

    if type(elements[0]) == list:
        outer = []
        for e, sh in zip(elements, shapes):
            outer.append(broadcast_val(e, sh))
        return outer
    else:
        return broadcast_val(elements, shapes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
