import logging
import statistics

from archai.supergraph.algos.nasbench101.nasbench101_dataset import Nasbench101Dataset


def main():
    logging.getLogger().setLevel(logging.DEBUG)

    # create dataset
    nsds= Nasbench101Dataset('~/dataroot/nasbench_ds/nasbench_full.pkl')

    vars = [statistics.variance(statistics.mean(nsds.get_test_acc(i, epochs=e)) for e in Nasbench101Dataset.VALID_EPOCHS) \
            for i in range(len(nsds))]

    bad_archs = list((i,v) for i,v in enumerate(vars) if v < 0.01)
    print(bad_archs)


if __name__ == '__main__':
    main()