import logging
import random
from archai.algos.nasbench101.nasbench101_dataset import Nasbench101Dataset
from archai.algos.nasbench101 import model_builder

import plotly.express as px

def main():

    # create dataset
    nsds= Nasbench101Dataset('~/dataroot/nasbench_ds/nasbench_full.pkl')

    test_accs = []

    for arch_id in range(len(nsds)):
        all_trials = nsds.get_test_acc(arch_id)
        test_accuracy = sum(all_trials) / len(all_trials)
        test_accs.append(test_accuracy)

    fig = px.histogram(test_accs, labels={'x': 'Test Accuracy', 'y': 'Counts'})
    fig.show()




if __name__ == '__main__':
    main()