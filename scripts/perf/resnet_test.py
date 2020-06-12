# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import torch
from archai import cifar10_models

from archai.common.trainer import Trainer
from archai.common.config import Config
from archai.common.common import logger, common_init
from archai.datasets import data

def train_test(conf_eval:Config):
    # region conf vars
    conf_loader       = conf_eval['loader']
    conf_trainer = conf_eval['trainer']
    # endregion

    conf_trainer['validation']['freq']=1
    conf_trainer['epochs'] = 10
    conf_loader['train_batch'] = 128
    conf_loader['test_batch'] = 4096
    conf_loader['cutout'] = 0
    conf_trainer['drop_path_prob'] = 0.0
    conf_trainer['grad_clip'] = 0.0
    conf_trainer['aux_weight'] = 0.0

    Net = cifar10_models.resnet34
    model = Net().to(torch.device('cuda', 0))

    # get data
    train_dl, _, test_dl = data.get_data(conf_loader)
    assert train_dl is not None and test_dl is not None

    trainer = Trainer(conf_trainer, model, None)
    trainer.fit(train_dl, test_dl)


if __name__ == '__main__':
    conf = common_init(config_filepath='confs/algos/darts.yaml',
                       param_args=['--common.experiment_name', 'restnet_test'])

    conf_eval = conf['nas']['eval']

    # evaluate architecture using eval settings
    train_test(conf_eval)

    exit(0)

