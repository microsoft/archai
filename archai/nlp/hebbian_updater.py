import numpy as np
from torch import nn
import torch



def hebbian_weight_update(c, net, hiddens, counters, temp_counters):
    with torch.no_grad():
        for i, (hidden_i, target_i) in hiddens.items():
            target_i = target_i.reshape(-1)
            c_i = counters[i]
            n_i = temp_counters[i]
            n_cls = c_i.size(0)
            lam = (1 / c_i.float()).clamp(min=c.hebbian_gamma) * (c_i < c.hebbian_T).float()

            n_i.index_add_(0, target_i, torch.ones_like(target_i))

            weight = (net.module if c.distributed else net).loss.layers[i].weight.data

            h_sums = torch.zeros_like(weight)
            hidden_i = hidden_i * weight[target_i].norm(dim=1, keepdim=True).to(hidden_i.dtype) / hidden_i.norm(dim=1, keepdim=True)
            h_sums.index_add_(0, target_i, hidden_i.to(h_sums.dtype))

            if c.distributed:
                all_h_sums = [torch.zeros_like(h_sums) for _ in range(c.world_size)]
                torch.distributed.all_gather(all_h_sums, h_sums)

                all_n_is = [torch.zeros_like(n_i) for _ in range(c.world_size)]
                torch.distributed.all_gather(all_n_is, n_i)
                h_sums = sum(all_h_sums)
                n_i = sum(all_n_is)

            c_i += n_i # update total count

            mask = n_i > 0
            h_sums = h_sums[mask]
            n_i = n_i[mask]

            h_means = lam[mask][:, None] * h_sums / n_i[:, None].to(h_sums.dtype) # divide by mean then scale by lambda

            weight[mask].mul_(1 - lam[mask][:, None]) # scale by 1 - lambda
            weight[mask].add_(h_means)

            n_i[:] = 0
