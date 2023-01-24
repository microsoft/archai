# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from typing import Dict, Type

from archai.supergraph.utils import utils
from archai.supergraph.nas.exp_runner import ExperimentRunner
from archai.supergraph.algos.darts.darts_exp_runner import DartsExperimentRunner
from archai.supergraph.algos.petridish.petridish_exp_runner import PetridishExperimentRunner
from archai.supergraph.algos.random.random_exp_runner import RandomExperimentRunner
from archai.supergraph.algos.manual.manual_exp_runner import ManualExperimentRunner
from archai.supergraph.algos.xnas.xnas_exp_runner import XnasExperimentRunner
from archai.supergraph.algos.gumbelsoftmax.gs_exp_runner import GsExperimentRunner
from archai.supergraph.algos.divnas.divnas_exp_runner import DivnasExperimentRunner
from archai.supergraph.algos.didarts.didarts_exp_runner import DiDartsExperimentRunner


def main():
    runner_types:Dict[str, Type[ExperimentRunner]] = {
        'darts': DartsExperimentRunner,
        'petridish': PetridishExperimentRunner,
        'xnas': XnasExperimentRunner,
        'random': RandomExperimentRunner,
        'manual': ManualExperimentRunner,
        'gs': GsExperimentRunner,
        'divnas': DivnasExperimentRunner,
        'didarts': DiDartsExperimentRunner
    }

    parser = argparse.ArgumentParser(description='NAS E2E Runs')
    parser.add_argument('--algos', type=str, default='darts,xnas,random,didarts,petridish,gs,manual,divnas',
                        help='NAS algos to run, seperated by comma')
    parser.add_argument('--datasets', type=str, default='cifar10',
                        help='datasets to use, separated by comma')
    parser.add_argument('--full', type=lambda x:x.lower()=='true',
                        nargs='?', const=True, default=False,
                        help='Run in full or toy mode just to check for compile errors')
    parser.add_argument('--no-search', type=lambda x:x.lower()=='true',
                        nargs='?', const=True, default=False,
                        help='Do not run search')
    parser.add_argument('--no-eval', type=lambda x:x.lower()=='true',
                        nargs='?', const=True, default=False,
                        help='Do not run eval')
    parser.add_argument('--exp-prefix', type=str, default='throwaway',
                        help='Experiment prefix is used for directory names')
    args, extra_args = parser.parse_known_args()

    if '--common.experiment_name' in extra_args:
        raise RuntimeError('Please use --exp-prefix instead of --common.experiment_name so that main.py can generate experiment directories with search and eval suffix')

    for dataset in args.datasets.split(','):
        for algo in args.algos.split(','):
            algo = algo.strip()
            print('Running (algo, dataset): ', (algo, dataset))
            runner_type:Type[ExperimentRunner] = runner_types[algo]

            # get the conf files for algo and dataset
            algo_conf_filepath = f'confs/algos/{algo}.yaml' if args.full \
                                               else f'confs/algos/{algo}_toy.yaml'
            dataset_conf_filepath = f'confs/datasets/{dataset}.yaml'
            conf_filepaths = ';'.join((algo_conf_filepath, dataset_conf_filepath))

            runner = runner_type(conf_filepaths,
                                base_name=f'{algo}_{dataset}_{args.exp_prefix}',
                                # for toy and debug runs, clean exp dirs
                                clean_expdir=utils.is_debugging() or not args.full)

            runner.run(search=not args.no_search, eval=not args.no_eval)


if __name__ == '__main__':
    main()
