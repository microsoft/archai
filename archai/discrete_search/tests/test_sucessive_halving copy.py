import torch
import time
from tqdm import tqdm

from archai.metrics.onnx_model import AvgOnnxLatencyMetric
from archai.metrics.torch_model import FlopsMetric, NumParametersMetric
from archai.metrics.functional import FunctionalMetric
from archai.metrics.ray import RayParallelMetric
from archai.metrics import evaluate_models, get_pareto_frontier

from archai.algos.sucessive_halving.sucessive_halving import SucessiveHalvingAlgo
from archai.search_spaces.discrete.natsbench_tss.search_space import NatsbenchTssSearchSpace
from archai.metrics.lookup import NatsBenchMetric

from archai.nas.arch_meta import ArchWithMetaData
from archai.datasets.dataset_provider import DatasetProvider


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
algo = SucessiveHalvingAlgo(
    ss, objectives, dataset_provider=None, num_iters=4,
    init_num_models=100, init_budget=1.0,
    output_dir='/home/pkauffmann/logdir/sucessive_halving3_natsbench'
)
algo.search()
