import logging
import random
from archai.algos.nasbench101.nasbench101_dataset import Nasbench101Dataset
from archai.algos.nasbench101 import model_builder
import statistics

def main():
    logging.getLogger().setLevel(logging.DEBUG)

    # create dataset
    nsds= Nasbench101Dataset('~/dataroot/nasbench_ds/nasbench_full.pkl')

    vars = [statistics.variance(statistics.mean(nsds.get_test_acc(i, epochs=e)) for e in Nasbench101Dataset.VALID_EPOCHS) \
            for i in range(len(nsds))]

    print(vars[:100])


if __name__ == '__main__':
    main()