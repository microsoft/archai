import torch
import time
from tqdm import tqdm

from archai.discrete_search.metrics.torch_model import FlopsMetric, NumParametersMetric

from archai.discrete_search.algos.evolution_pareto import EvolutionParetoSearch
from archai.discrete_search.search_spaces.natsbench_tss.search_space import NatsbenchTssSearchSpace
from archai.discrete_search.metrics.lookup import NatsBenchMetric


natsbench_path = '/home/pkauffmann/dataroot/natsbench/NATS-tss-v1_0-3ffb9-simple/'

# Segmentation search space
ss = NatsbenchTssSearchSpace(natsbench_path, 'cifar100')

# Objective list
objectives = {
    'Flops': FlopsMetric(input_shape=(1, 3, 32, 32)),

    'Test Accuracy with 2 epochs of training (%)': NatsBenchMetric(
        ss,
        metric_name='test-accuracy', higher_is_better=True,
        epochs=2
    )
}

# Search
algo = EvolutionParetoSearch(
    ss, objectives, dataset_provider=None,
    num_crossovers=0, mutations_per_parent=10, init_num_models=30, num_random_mix=10, num_iters=15,
    output_dir='/home/pkauffmann/logdir/evolutionary_search_test_natsbench'
)
algo.search()

print(f'Total training time spent = {objectives["Test Accuracy with 2 epochs of training (%)"].total_time_spent}')
