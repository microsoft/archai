import logging
import random

from archai.supergraph.algos.nasbench101 import model_builder
from archai.supergraph.algos.nasbench101.nasbench101_dataset import Nasbench101Dataset


def main():
    logging.getLogger().setLevel(logging.DEBUG)

    # create dataset
    nsds = Nasbench101Dataset("~/dataroot/nasbench_ds/nasbench_full.pkl")

    # create model by index
    model = nsds.create_model(42)
    print(model)

    model4 = nsds[4]
    print("model4", model4)

    # query for specific model
    data = nsds.query(model_builder.EXAMPLE_DESC_MATRIX, model_builder.EXAMPLE_VERTEX_OPS)
    print("queried", data)

    # sample model
    # nsds is list type object of model statistics
    num_models = len(nsds)
    data = nsds[random.randint(0, num_models - 1)]
    print("random", data)

    # nsds is pre-sorted by avg test accuracy
    print("worst acc", nsds.get_test_acc(0))
    print("worst acc uninit", nsds.get_test_acc(0, step_index=0))
    print("best acc", nsds.get_test_acc(len(nsds) - 1))
    print("best acc uninit", nsds.get_test_acc(len(nsds) - 1, step_index=0))
    print("best acc epoch 4", nsds.get_test_acc(len(nsds) - 1, epochs=4))
    print("best acc epoch 36", nsds.get_test_acc(len(nsds) - 1, epochs=36))


if __name__ == "__main__":
    main()
