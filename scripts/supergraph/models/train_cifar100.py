# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from archai.common.config import Config
from archai.supergraph import models
from archai.supergraph.datasets import data
from archai.common.common import common_init
from archai.supergraph.utils.trainer import Trainer


def train_test(conf_eval: Config):
    conf_loader = conf_eval["loader"]
    conf_trainer = conf_eval["trainer"]

    # create model
    Net = models.resnet34
    model = Net().to(torch.device("cuda", 0))

    # get data
    data_loaders = data.get_data(conf_loader)

    # train!
    trainer = Trainer(conf_trainer, model)
    trainer.fit(data_loaders)


if __name__ == "__main__":
    conf = common_init(config_filepath="confs/algos/resnet.yaml;confs/datasets/cifar100.yaml")
    conf_eval = conf["nas"]["eval"]

    train_test(conf_eval)
