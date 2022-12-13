import argparse

import azureml.core
from azureml.telemetry import set_diagnostics_collection
from azureml.core.workspace import Workspace
from azureml.core import Datastore
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Experiment
from azureml.core.container_registry import ContainerRegistry
from azureml.train.estimator import Estimator
from azureml.core import Environment

from archai.common.config import Config
from tools.azure.aml_experiment_runner import AmlExperimentRunner


def main():

    runner_types = ('darts', 'petridish', 'xnas', 'random')

    parser = argparse.ArgumentParser(description='NAS E2@ Runs on AML')
    parser.add_argument('--aml_secrets_filepath', type=str, help='Path to AML secrets file in the same template as aml_config_template.yaml')
    parser.add_argument('--algo', type=str, help=f'NAS algo to run. Has to be one of {runner_types}')    
    parser.add_argument('--full', action='store_true', default=False,
                        help='Run in full or toy mode just to check for compile errors')
    parser.add_argument('--no-search', action='store_true', default=False,
                        help='Run search')
    parser.add_argument('--no-eval', action='store_true', default=False,
                        help='Run eval')
    parser.add_argument('--exp-prefix', type=str, default='throwaway',
                        help='Experiment prefix is used for directory names')
    args, extra_args = parser.parse_known_args()

    # sanity check
    if args.algo not in runner_types:
        raise NotImplementedError

    # create aml experiment runner
    aml_runner = AmlExperimentRunner(args.aml_secrets_filepath)
    input_ds = aml_runner.input_datastore_handle
    output_ds = aml_runner.output_datastore_handle

    # create script params
    script_params = {'--nas.eval.loader.dataset.dataroot': input_ds.path('/').as_mount(),
                     '--nas.search.loader.dataset.dataroot': input_ds.path('/').as_mount(),
                     '--common.logdir': output_ds.path('/').as_mount(),
                     '--exp-prefix': args.exp_prefix,
                     '--algos': args.algo,
                    }

    if args.full:
        script_params['--full'] = ''
    if args.no_search:
        script_params['--no-search'] = ''
    if args.no_eval:
        script_params['--no-eval'] = ''
    
    # create entry script
    entry_script = 'scripts/main.py'

    # launch experiment
    aml_runner.launch_experiment(args.algo, script_params, entry_script)
    print('Finished launching job')

    
if __name__ == '__main__':
    main()










