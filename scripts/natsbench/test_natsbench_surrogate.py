from pathlib import Path
from argparse import ArgumentParser

import torch

from archai.search_spaces.discrete_search_spaces.natsbench_tss_search_spaces.discrete_search_space_natsbench_tss import DiscreteSearchSpaceNatsbenchTSS
from archai.nas.surrogate.predictive_dnn_ensemble import PredictiveDNNEnsemble
from archai.nas.encoders.path_encoder import PathEncoder

parser = ArgumentParser('Encodes Natsbench TSS architectures using PathEncoder or path n-grams')
parser.add_argument('--use_first', type=int, default=500)
parser.add_argument('--tr_size', type=float, default=0.8)
parser.add_argument('--encoder', type=str, choices=['path', 'path_ngram', 'gcn'], default='path')
parser.add_argument('--ngram', type=int, default=3)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--nats_dir', type=Path, default=Path.home() / 'dataroot/natsbench/NATS-tss-v1_0-3ffb9-simple/')


if __name__ == '__main__':
    args = parser.parse_args()
    args.ngram = -1 if args.encoder == 'path' else args.ngram

    # Gets architecture graphs and test accuracy
    sp = DiscreteSearchSpaceNatsbenchTSS(args.dataset, str(args.nats_dir))
    n = args.use_first if args.use_first else len(sp)
    models = [sp.get_arch_repr_from_archstr(archstr) for i, archstr in enumerate(sp.api) if i <= n]
    acc = [sp.api.get_more_info(i, args.dataset)['test-accuracy']/100 for i, _ in enumerate(models)]

    split = int(len(models) * args.tr_size)
    tr_X, tr_y, te_X, te_y = models[:split], acc[:split], models[split:], acc[split:]

    # Fits featurizer
    pe = PathEncoder(path_length=-1)
    tr_Xt = torch.tensor(pe.fit_transform(tr_X), dtype=torch.float)
    te_Xt = torch.tensor(pe.transform(te_X), dtype=torch.float)

    tr_y, te_y = [torch.tensor(y, dtype=torch.float)[..., None] for y in [tr_y, te_y]]

    # Trains neural predictor
    surrogate = PredictiveDNNEnsemble(num_features=tr_Xt.shape[1], num_ensemble_members=5, sigmoid=True)
    surrogate.to_cuda()

    surrogate.fit(tr_Xt.cuda(), tr_y.cuda(), {'lr': 1e-4, 'num_steps': 1_000})
    te_pred = [surrogate.predict(te_Xt[i][None, ...].cuda()) for i in range(te_Xt.shape[0])]

    print(te_pred)
