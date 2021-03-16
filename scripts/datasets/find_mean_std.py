# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from archai.common.config import Config
from archai.common.common import default_dataroot
from archai.datasets import data
from torchvision import transforms
from archai.common.ml_utils import channel_norm

if __name__ == '__main__':

    # create env vars that might be used in paths in config
    if 'default_dataroot' not in os.environ:
        os.environ['default_dataroot'] = default_dataroot()

    conf = Config(config_filepath='confs/datasets/synthetic_cifar10.yaml')

    conf_dataset = conf['dataset']

    ds_provider = data.create_dataset_provider(conf_dataset)

    # you have to manually type in the transform here since you
    # don't know the STD and MEAN values
    train_ds, _ = ds_provider.get_datasets(True, False,
        transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
        transforms.Compose([]))

    print(channel_norm(train_ds))

    exit(0)