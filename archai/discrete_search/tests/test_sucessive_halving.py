import torch
import time
from tqdm import tqdm

from archai.discrete_search.metrics.onnx_model import AvgOnnxLatencyMetric
from archai.discrete_search.metrics.ray import RayParallelMetric

from archai.discrete_search.algos.sucessive_halving import SucessiveHalvingSearch
from archai.discrete_search.search_spaces.natsbench_tss.search_space import NatsbenchTssSearchSpace
from archai.discrete_search.metrics.lookup import NatsBenchMetric

from archai.discrete_search import SearchResults, NasModel


natsbench_path = '/home/pkauffmann/dataroot/natsbench/NATS-tss-v1_0-3ffb9-simple/'

# Segmentation search space
ss = NatsbenchTssSearchSpace(natsbench_path, 'cifar100')

# Objective list
objectives = {
    'ONNX Latency': RayParallelMetric(AvgOnnxLatencyMetric(input_shape=(1, 3, 64, 64)), num_cpus=1.0),
    'Test accuracy': NatsBenchMetric(ss, metric_name='test-accuracy', higher_is_better=True
    )
}

# Search
algo = SucessiveHalvingSearch(
    ss, objectives, dataset_provider=None, num_iters=4,
    init_num_models=100, init_budget=1.0,
    output_dir='/home/pkauffmann/logdir/sucessive_halving3_natsbench'
)
algo.search()
