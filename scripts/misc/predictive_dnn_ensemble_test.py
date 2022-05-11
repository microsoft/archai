import torch
import torch.nn as nn

from archai.nas.predictive_dnn_ensemble import PredictiveDNNEnsemble


def main():
    ensemble = PredictiveDNNEnsemble(num_ensemble_members=5)

    # create some data
    feat_dim = 128
    num_points = 1024

    x = torch.rand(num_points, feat_dim)
    y = torch.rand(num_points, 1)

    conf_train = dict(lr=0.0001, num_steps=20000)

    # fit
    ensemble.fit(x=x, y=y, conf_train=conf_train)

    print('dummy')

    















if __name__ == '__main__':
    main()