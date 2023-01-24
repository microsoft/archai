# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time

import ray
import torch

from archai.supergraph import models
from archai.supergraph.utils.trainer import Trainer
from archai.supergraph.utils import common
from archai.supergraph.datasets import data
from archai.supergraph.utils.metrics import Metrics

def train_test()->Metrics:
    conf = common.get_conf()
    conf_eval = conf['nas']['eval']

    # region conf vars
    conf_loader       = conf_eval['loader']
    conf_trainer = conf_eval['trainer']
    # endregion

    conf_trainer['validation']['freq']=1
    conf_trainer['epochs'] = 1
    conf_loader['train_batch'] = 128
    conf_loader['test_batch'] = 4096
    conf_loader['cutout'] = 0
    conf_trainer['drop_path_prob'] = 0.0
    conf_trainer['grad_clip'] = 0.0
    conf_trainer['aux_weight'] = 0.0

    Net = models.resnet34
    model = Net().to(torch.device('cuda'))

    # get data
    data_loaders = data.get_data(conf_loader)
    assert data_loaders.train_dl is not None and data_loaders.test_dl is not None

    trainer = Trainer(conf_trainer, model, None)
    trainer.fit(data_loaders)
    met = trainer.get_metrics()
    return met

@ray.remote(num_gpus=1)
def train_test_ray(common_state):
    common.init_from(common_state)
    return train_test()

def train_test_dist():
    start = time.time()
    result_ids = [train_test_ray.remote(common.get_state()) for x in range(2)]
    while len(result_ids):
        done_id, result_ids = ray.wait(result_ids)
        metrics:Metrics = ray.get(done_id[0])
        print(f'result={metrics.run_metrics.epochs_metrics[-1].top1.avg}, '
              f'time={time.time()-start}')

if __name__ == '__main__':
    ray.init(num_gpus=1)
    print('ray init done')
    common.common_init(config_filepath='confs/algos/darts.yaml',
                       param_args=['--common.experiment_name', 'resnet_test'])

    train_test_dist()

    exit(0)