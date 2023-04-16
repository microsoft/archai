# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys
import itertools
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Optional


from archai.common.config import Config
from archai.common.store import ArchaiStore
from archai.datasets.cv.face_synthetics import FaceSyntheticsDatasetProvider
from archai.discrete_search.api import SearchObjectives
from archai.discrete_search.algos import (
    MoBananasSearch, EvolutionParetoSearch, LocalSearch,
    RandomSearch, RegularizedEvolutionSearch
)
from archai.discrete_search.evaluators import TorchNumParameters, AvgOnnxLatency, RayParallelEvaluator
from archai.discrete_search.evaluators.remote_azure_benchmark import RemoteAzureBenchmarkEvaluator

from search_space.hgnet import HgnetSegmentationSearchSpace
from training.partial_training_evaluator import PartialTrainingValIOU
from training.aml_training_evaluator import AmlPartialTrainingValIOU
from utils.setup import configure_store

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


def filter_extra_args(extra_args: List[str], prefix: str) -> List[str]:
    return list(itertools.chain([
        [arg, val]
        for arg, val in zip(extra_args[::2], extra_args[1::2])
        if arg.startswith(prefix)
    ]))


def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, help='Face Synthetics dataset directory.')
    parser.add_argument('--output_dir', type=Path, help='Output directory.', default='output')
    parser.add_argument('--search_config', type=Path, help='Search config file.', default=confs_path / 'cpu_search.yaml')
    parser.add_argument('--serial_training', help='Search config file.', action='store_true')
    parser.add_argument('--gpus_per_job', type=float, help='Number of GPUs used per job (if `serial_training` flag is disabled)',
                        default=0.5)
    parser.add_argument('--partial_tr_epochs', type=int, help='Number of epochs to run partial training', default=1)
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--timeout', type=int, help='Timeout for partial training (in seconds)(default 10800)', default=10800)

    args, extra_args = parser.parse_known_args()

    timeout_seconds = args.timeout

    # Filters extra args that have the prefix `search_space`
    search_extra_args = filter_extra_args(extra_args, 'search.')
    config = Config(str(args.search_config), search_extra_args, resolve_env_vars=True)
    search_config = config['search']

    # Search space
    ss_config = search_config['search_space']

    search_space = AVAILABLE_SEARCH_SPACES[ss_config['name']](
        seed=args.seed,
        **ss_config.get('params', {}),
    )

    input_shape = (1, search_space.in_channels, *search_space.img_size[::-1])

    partial_training_output = args.output_dir / 'partial_training_logs'
    os.makedirs(partial_training_output, exist_ok=True)

    # Search objectives
    so = SearchObjectives()

    target_config = search_config.get('target', {})
    target_name = target_config.pop('name', 'cpu')
    assert target_name in ['cpu', 'snp', 'aml']

    max_latency = 0.3 if target_name == 'cpu' else 0.185
    algo_config = search_config['algorithm']
    algo_params = algo_config.get('params', {})
    max_parameters = float(algo_params.pop('max_parameters', 5e7))

    # Adds a constraint on number of parameters so we don't sample models that are too large
    so.add_constraint(
        'Model Size (b)',
        TorchNumParameters(),
        constraint=(1e6, max_parameters)
    )

    # Adds a constrained objective on model latency so we don't pick models that are too slow.
    so.add_objective(
        'CPU ONNX Latency (s)',
        AvgOnnxLatency(
            input_shape=input_shape, export_kwargs={'opset_version': 11}
        ),
        higher_is_better=False,
        compute_intensive=False,
        constraint=[0, max_latency]
    )

    if target_name == 'snp' or target_name == 'aml':
        # Gets connection string from env variable
        aml_config = config['aml']
        experiment_name = aml_config.get('experiment_name', 'facesynthetics')
        store: ArchaiStore = configure_store(aml_config)

        evaluator = RemoteAzureBenchmarkEvaluator(
            input_shape=input_shape,
            store=store,
            experiment_name=experiment_name,
            onnx_export_kwargs={'opset_version': 11},
            **target_config
        )

        so.add_objective(
            'SNP Quantized Latency (s)',
            evaluator,
            higher_is_better=False,
            compute_intensive=True
        )

    if target_name == 'aml':
        # do the partial training in an AML gpu cluster
        partial_tr_obj = AmlPartialTrainingValIOU(
            config,
            tr_epochs=int(args.partial_tr_epochs),
            timeout_seconds=timeout_seconds,
            local_output=partial_training_output
        )

    else:
        if args.dataset_dir is None:
            raise ValueError('--dataset_dir must be specified if target is not aml')

        # Dataset provider
        dataset_provider = FaceSyntheticsDatasetProvider(args.dataset_dir)

        partial_tr_obj = PartialTrainingValIOU(
            dataset_provider,
            tr_epochs=args.partial_tr_epochs,
            output_dir=partial_training_output
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

    # Search algorithm
    algo = AVAILABLE_ALGOS[algo_config['name']](
        search_space, so,
        output_dir=args.output_dir,
        seed=args.seed,
        **algo_params,
    )

    algo.search()


if __name__ == '__main__':
    main()
