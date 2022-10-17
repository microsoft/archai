import torch
import time
from tqdm import tqdm

from archai.discrete_search.metrics.onnx_model import AvgOnnxLatencyMetric
from archai.discrete_search.metrics.torch_model import FlopsMetric, NumParametersMetric
from archai.discrete_search.metrics.functional import FunctionalMetric
from archai.discrete_search.metrics.ray import RayParallelMetric
from archai.discrete_search.metrics import evaluate_models

from archai.discrete_search.algos.bananas import MoBananasSearch
from archai.discrete_search.search_spaces.natsbench_tss.search_space import NatsbenchTssSearchSpace
from archai.discrete_search.metrics.lookup import NatsBenchMetric


natsbench_path = '/home/pkauffmann/dataroot/natsbench/NATS-tss-v1_0-3ffb9-simple/'

# Segmentation search space
ss = NatsbenchTssSearchSpace(natsbench_path, 'cifar100')

# Objective list
objectives = {
    'ONNX Latency': RayParallelMetric(AvgOnnxLatencyMetric(input_shape=(1, 3, 64, 64)), num_cpus=1.0),
    #'Number of parameters': NumParametersMetric(input_shape=(1, 3, 64, 64)),
    'Test accuracy': NatsBenchMetric(ss, metric_name='test-accuracy', higher_is_better=True)
}

# Search
algo = MoBananasSearch(
    '/home/pkauffmann/logdir/bananas_natsbench',
    ss, objectives, dataset_provider=None,
    cheap_objectives=['ONNX Latency'],
    num_iters=10, num_parents=20, mutations_per_parent=2,
    num_mutations=20, init_num_models=10
)
algo.search()
