import os
import itertools
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Optional


from archai.common.config import Config
from archai.datasets.cv.face_synthetics import FaceSyntheticsDatasetProvider
from archai.discrete_search.api import SearchObjectives
from archai.discrete_search.algos import (
    MoBananasSearch, EvolutionParetoSearch, LocalSearch,
    RandomSearch, RegularizedEvolutionSearch
)
from archai.discrete_search.evaluators.onnx_model import AvgOnnxLatency
from archai.discrete_search.evaluators.remote_azure_benchmark import RemoteAzureBenchmarkEvaluator
from archai.discrete_search.evaluators.ray import RayParallelEvaluator

from search_space.hgnet import HgnetSegmentationSearchSpace
from training.partial_training_evaluator import PartialTrainingValIOU

AVAILABLE_ALGOS = {
    'mo_bananas': MoBananasSearch,
    'evolution_pareto': EvolutionParetoSearch,
    'local_search': LocalSearch,
    'random_search': RandomSearch,
    'regularized_evolution': RegularizedEvolutionSearch
}

AVAILABLE_SEARCH_SPACES = {
    'hgnet': HgnetSegmentationSearchSpace,
}

confs_path = Path(__file__).absolute().parent / 'confs'

parser = ArgumentParser()
parser.add_argument('--dataset_dir', type=Path, help='Face Synthetics dataset directory.', required=True)
parser.add_argument('--output_dir', type=Path, help='Output directory.', required=True)
parser.add_argument('--search_config', type=Path, help='Search config file.', default=confs_path / 'evolution_pareto_config.yaml')
parser.add_argument('--serial_training', help='Search config file.', action='store_true')
parser.add_argument('--gpus_per_job', type=float, help='Number of GPUs used per job (if `serial_training` flag is disabled)',
                    default=0.5)
parser.add_argument('--partial_tr_epochs', type=float, help='Number of epochs to run partial training', default=1.0)
parser.add_argument('--seed', type=int, help='Random seed', default=42)


def filter_extra_args(extra_args: List[str], prefix: str) -> List[str]:
    return list(itertools.chain([
        [arg, val]
        for arg, val in zip(extra_args[::2], extra_args[1::2])
        if arg.startswith(prefix)
    ]))


if __name__ == '__main__':
    args, extra_args = parser.parse_known_args()

    # Filters extra args that have the prefix `search_space`
    search_extra_args = filter_extra_args(extra_args, 'search.')
    search_config = Config(str(args.search_config), search_extra_args)['search']

    # Search space
    ss_config = search_config['search_space']

    search_space = AVAILABLE_SEARCH_SPACES[ss_config['name']](
        seed=args.seed,
        **ss_config.get('params', {}),
    )

    input_shape = (1, search_space.in_channels, *search_space.img_size[::-1])

    # Search objectives
    so = SearchObjectives()

    target_config = search_config.get('target', {})
    target_name = target_config.pop('name', 'cpu')
    assert target_name in ['cpu', 'snp']

    if target_name == 'cpu':
        so.add_objective(
            'CPU ONNX Latency (s)',
            AvgOnnxLatency(
                input_shape=input_shape, export_kwargs={'opset_version': 11}
            ),
            higher_is_better=False,
            compute_intensive=False,
            constraint=[0, 0.3]
        )
    else:
        # Gets connection string from env variable
        target_config['connection_string'] = os.environ[
            target_config.pop('connection_str_env_var')
        ]

        evaluator = RemoteAzureBenchmarkEvaluator(
            input_shape=input_shape, 
            onnx_export_kwargs={'opset_version': 11},
            **target_config
        )

        # Adds the same constraint so we don't sample models that are too large
        so.add_constraint(
            'CPU ONNX Latency (s)',
            AvgOnnxLatency(
                input_shape=input_shape, export_kwargs={'opset_version': 11}
            ),
            constraint=[0, 0.185]
        )

        so.add_objective(
            'SNP Quantized Latency (s)',
            evaluator,
            higher_is_better=False,
            compute_intensive=True
        )

    partial_tr_obj = PartialTrainingValIOU(
        tr_epochs=args.partial_tr_epochs,
        output_dir=args.output_dir / 'partial_training_logs'
    )

    if not args.serial_training:
        partial_tr_obj = RayParallelEvaluator(
            partial_tr_obj, num_gpus=args.gpus_per_job, 
            max_calls=1
        )

    so.add_objective(
        'Partial Training Val. IOU',
        partial_tr_obj,
        higher_is_better=True,
        compute_intensive=True
    )

    # Dataset provider
    dataset_provider = FaceSyntheticsDatasetProvider(args.dataset_dir)

    # Search algorithm
    algo_config = search_config['algorithm']
    algo = AVAILABLE_ALGOS[algo_config['name']](
        search_space, so, dataset_provider, 
        output_dir=args.output_dir, seed=args.seed,
        **algo_config.get('params', {}),
    )

    algo.search()
