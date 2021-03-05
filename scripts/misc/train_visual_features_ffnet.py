# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from archai.networks.visual_features_with_ff_net import VisualFeaturesWithFFNet
from archai.common.trainer import Trainer
from archai.common.config import Config
from archai.common.common import common_init
from archai.datasets import data

def train_test(conf_eval:Config):
    conf_loader = conf_eval['loader']
    conf_trainer = conf_eval['trainer']

    # create model
    Net = VisualFeaturesWithFFNet
    feature_len = 324
    n_classes = 10
    model = Net(feature_len, n_classes).to(torch.device('cuda',  0))

    # get data
    data_loaders = data.get_data(conf_loader)

    # train!
    trainer = Trainer(conf_trainer, model)
    trainer.fit(data_loaders)


if __name__ == '__main__':
    conf = common_init(config_filepath='confs/algos/visual_features_ffnet.yaml')
    conf_eval = conf['nas']['eval']

    train_test(conf_eval)


