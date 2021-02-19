import numpy as np
from torch import nn
import torch

class AdaptiveEmbedding(nn.Module):
    def __init__(self, c):
        super(AdaptiveEmbedding, self).__init__()
        self.c = c.setdefault(cutoffs=[], adaptive_ratio=1)

        if c.get('n_embeds'):
            n_embeds = c.n_embeds
        else:
            c.n_embeds = n_embeds = [c.n_embed // (c.adaptive_ratio ** i) for i in range(len(c.cutoffs) + 1)]
        assert n_embeds[0] == c.n_embed

        self.layers = nn.ModuleList(
            nn.Embedding(end - start, n_embed_i) for n_embed_i, start, end in zip(
                n_embeds, [0] + c.cutoffs, c.cutoffs + [c.n_vocab]
            )
        )
        self.projections = nn.ModuleList(
            nn.Linear(n_embed_i, c.n_embed, bias=False) for n_embed_i in n_embeds[1:]
        )
        for layer in self.layers:
            nn.init.normal_(layer.weight, 0, 0.02)

    def forward(self, x):
        c = self.c
        x_flat = x.reshape(-1)

        emb_flat = torch.zeros([x_flat.size(0), c.n_embed], dtype=torch.float if c.opt_level == 'O0' else torch.half, device=x.device)
        for i, (layer, start, end) in enumerate(zip(self.layers, [0] + c.cutoffs, c.cutoffs + [c.n_vocab])):
            mask_i = (x_flat >= start) & (x_flat < end)
            indices_i = mask_i.nonzero().squeeze()

            if indices_i.numel() == 0:
                continue

            inp_i = x_flat.index_select(0, indices_i) - start
            emb_i = layer(inp_i)
            if i == 0:
                if c.opt_level != 'O0':
                    emb_i = emb_i.half()
            else:
                emb_i = self.projections[i - 1](emb_i)
            emb_flat.index_copy_(0, indices_i, emb_i)

        emb = emb_flat.view(*x.size(), c.n_embed)
        emb.mul_(np.sqrt(c.n_embed))
        return emb
