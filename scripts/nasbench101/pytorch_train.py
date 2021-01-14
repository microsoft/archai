from archai.algos.nasbench101 import model_builder
from archai import cifar10_models
from archai.common.trainer import Trainer
from archai.common.config import Config
from archai.common.common import common_init
from archai.datasets import data


def main():
    conf = common_init(config_filepath='confs/algos/nasbench101.yaml')
    conf_eval = conf['nas']['eval']
    conf_loader = conf_eval['loader']
    conf_trainer = conf_eval['trainer']

    model = model_builder.build(model_builder.EXAMPLE_DESC_MATRIX, model_builder.EXAMPLE_VERTEX_OPS)

    train_dl, _, test_dl = data.get_data(conf_loader)

    trainer = Trainer(conf_trainer, model)
    trainer.fit(train_dl, test_dl)

if __name__ == '__main__':
    main()