# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from archai.networks.visual_features_with_ff_net import VisualFeaturesWithFFNet
from archai.common.trainer import Trainer
from archai.common.config import Config
from archai.common.common import common_init
from archai.datasets import data

def train_test(conf_eval:Config, conf_dataset:Config):
    conf_loader = conf_eval['loader']
    conf_trainer = conf_eval['trainer']

    # ----conf region----
    feature_len = conf_eval['feature_len']
    pixels_per_hog_cell = conf_eval['pixels_per_hog_cell']
    n_classes = conf_dataset['n_classes']
    # -------------------

    # create model    
    model = VisualFeaturesWithFFNet(feature_len, 
                                    n_classes, 
                                    pixels_per_hog_cell=(pixels_per_hog_cell, pixels_per_hog_cell)).to(torch.device('cuda',  0))

    # get data
    data_loaders = data.get_data(conf_loader)

    # train!
    trainer = Trainer(conf_trainer, model)
    trainer.fit(data_loaders)


if __name__ == '__main__':
    conf = common_init(config_filepath='confs/algos/visual_features_ffnet.yaml')
    conf_eval = conf['nas']['eval']
    conf_dataset = conf['dataset']

    train_test(conf_eval, conf_dataset)


