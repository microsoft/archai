import logging

from archai.algos.nasbench101.nasbench101_dataset import Nasbench101Dataset
from archai.algos.nasbench101 import model_builder
from archai import cifar10_models
from archai.common.trainer import Trainer
from archai.common.config import Config
from archai.common.common import common_init
from archai.datasets import data
from archai.algos.nasbench101.nasbench101_dataset import Nasbench101Dataset


def main():
    #6, 7, 9, 10, 16

    #model = model_builder.build(model_builder.EXAMPLE_DESC_MATRIX, model_builder.EXAMPLE_VERTEX_OPS)
    nsds= Nasbench101Dataset('~/dataroot/nasbench_ds/nasbench101_sample.tfrecord.pkl')
    for i in range(len(nsds)):
        conf = common_init(config_filepath='confs/algos/nasbench101.yaml')
        conf_eval = conf['nas']['eval']
        conf_loader = conf_eval['loader']
        conf_trainer = conf_eval['trainer']

        params = nsds[i]['trainable_parameters']
        if params < 10e6 or params > 15e6:
            continue
        print('selected:',i,params)
        model = nsds.create_model(i) # 401277 is same model as example

        train_dl, _, test_dl = data.get_data(conf_loader)

        trainer = Trainer(conf_trainer, model)
        trainer.fit(train_dl, test_dl)

if __name__ == '__main__':
    main()