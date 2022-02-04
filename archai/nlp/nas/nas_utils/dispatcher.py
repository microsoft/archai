# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Auxiliary methods that dispatches traning/evaluation jobs.
"""

import copy
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from archai.common import utils
from archai.nlp.nas.nas_utils.converter import Converter


def create_batch_jobs(configs: List[Dict[str, Any]],
                      default_config_file: str,
                      model_type: str,
                      max_step: int,
                      n_gpus: int,
                      gpu_config: str,
                      is_pareto: Optional[bool] = None) -> str:
    """Creates a batch of command-line jobs.

    Args:
        configs: List of configuration dictionaries.
        default_config_file: Default configuration file.
        model_type: Type of model.
        max_step: Maximum number of training steps.
        n_gpus: Number of GPUs.
        gpu_config: GPU configuration to be used.
        is_pareto: Whether job is from Pareto-frontier or not.

    Returns:
        (str): Command-line command.

    """

    command = []

    for i, config in enumerate(configs):
        # Makes sure that d_embed and d_head are defined properly
        config['d_embed'] = config['d_model']
        config['d_head'] = [config['d_model'] // n_head for n_head in config['n_head']]
        
        # Maps incoming configuration values to a string-based representation
        n_layer = config['n_layer']
        config_line = ''

        for key, value in config.items():
            if isinstance(value, list):
                str_value = ' '.join(str(v) for v in value[:n_layer])
            else:
                str_value = str(value)
            config_line += f' --{key} {str_value}'

        # Checks whether job comes from pareto points, which should have a specific identifier
        is_pareto_str = '_pareto' if (is_pareto is not None and is_pareto[i]) else ''
        exp_name = f'j_{i}{is_pareto_str}'

        # Creates the command-line
        line = f'python -m torch.distributed.launch --nproc_per_node={n_gpus} archai/nlp/train.py --model_type {model_type} --config {gpu_config} ' \
               f'--config_file {default_config_file} --max_step {max_step} --experiment_name {exp_name}' \
               f'{config_line}'
        
        command.append(line)

    return command


def create_jobs(configs: List[Dict[str, Any]],
                default_config_file: Optional[str] = 'wt103_base.yaml',
                model_type: Optional[str] = 'mem_transformer',
                max_step: Optional[int] = 500,
                start_config: Optional[int] = 0,
                n_jobs: Optional[int] = 50,
                n_gpus: Optional[int] = 8,
                gpu_config: Optional[str] = 'dgx1_8gpu_fp32',
                is_pareto: Optional[bool] = None,
                output_path: Optional[str] = '~/configs') -> None:
    """Creates command-line jobs.

    Args:
        configs: List of configurations.
        default_config_file: Default configuration file.
        model_type: Type of model.
        max_step: Number of maximum steps to train the models.
        start_config: Starting range of the configuration to be checked.
        n_jobs: Number of jobs to be created.
        n_gpus: Number of GPUs to be used.
        gpu_config: GPU configuration.
        is_pareto: Whether job is from Pareto-frontier or not.
        output_path: Save folder for the created command-lines.

    """

    output_config_path = utils.full_path(output_path, create=True)
    n_configs = len(configs)
    config_idx = start_config

    c = 0
    while c < n_configs:
        jobs_config = {}
        jobs_config['jobs'] = [{}]

        if is_pareto is not None:
            jobs_config['jobs'][0]['command'] = create_batch_jobs(copy.deepcopy(configs[c:c+n_jobs]), default_config_file, model_type, max_step, n_gpus, gpu_config, is_pareto=is_pareto[c:c+n_jobs])
        else:
            jobs_config['jobs'][0]['command'] = create_batch_jobs(copy.deepcopy(configs[c:c+n_jobs]), default_config_file, model_type, max_step, n_gpus, gpu_config)

        output_config_file = os.path.join(output_config_path, f'train_{config_idx}.yaml')
        with open(output_config_file, 'w') as f:
            yaml.dump(jobs_config, f)

        c += n_jobs
        config_idx += 1


def create_pareto_jobs(results_path: str, 
                       converter: Converter,
                       default_config_file: Optional[str] = 'wt103_base.yaml',
                       model_type: Optional[str] = 'mem_transformer',
                       max_step: Optional[int] = 500,
                       start_config: Optional[int] = 0,
                       n_jobs: Optional[int] = 50,
                       n_gpus: Optional[int] = 8,
                       gpu_config: Optional[str] = 'dgx1_8gpu_fp32',                             
                       output_path: Optional[str] = '~/configs') -> None:
    """Prepares command-line for training Pareto-frontier jobs.

    Args:
        results_path: Path to search results.
        converter: Converter class object.
        default_config_file: Default configuration file.
        model_type: Type of model.
        max_step: Number of maximum steps to train the models.
        start_config: Starting range of the configuration to be checked.
        n_jobs: Number of jobs to be created.
        n_gpus: Number of GPUs to be used.
        gpu_config: GPU configuration.
        output_path: Save folder for the created command-lines.

    """

    logs_path = os.path.join(results_path, 'logs.pkl')
    
    with open(logs_path, 'rb') as f:
        logs = pickle.load(f)

    # Gathers a list of pareto points
    pareto_pop = logs['pareto'][-1]['population']

    seen = set()
    pareto_configs = []
    is_pareto = []

    for gene in pareto_pop:
        model_key = converter.gene_to_key(gene)
        model_config = converter.gene_to_config(gene)

        if model_key in seen:
            continue

        else:
            seen.add(model_key)
            pareto_configs.append(model_config)
            is_pareto.append(True)

    create_jobs(pareto_configs,
                default_config_file=default_config_file,
                model_type=model_type,
                max_step=max_step,
                start_config=start_config,
                n_jobs=n_jobs,
                n_gpus=n_gpus,
                gpu_config=gpu_config, 
                is_pareto=is_pareto,
                output_path=output_path)


def create_ground_truth_jobs(results_path: str,
                             converter: Converter,
                             default_config_file: Optional[str] = 'wt103_base.yaml',
                             model_type: Optional[str] = 'mem_transformer',
                             max_step: Optional[int] = 500,
                             start_config: Optional[int] = 0,
                             n_jobs: Optional[int] = 50,
                             n_gpus: Optional[int] = 8,
                             gpu_config: Optional[str] = 'dgx1_8gpu_fp32',                             
                             output_path: Optional[str] = '~/configs') -> None:
    """Prepares command-lines for training all visited points during search.

    Args:
        results_path: Path to search results.
        converter: Converter class object.
        default_config_file: Default configuration file.
        model_type: Type of model.
        max_step: Number of maximum steps to train the models.
        start_config: Starting range of the configuration to be checked.
        n_jobs: Number of jobs to be created.
        n_gpus: Number of GPUs to be used.
        gpu_config: GPU configuration.
        output_path: Save folder for the created command-lines.

    """

    logs_path = os.path.join(results_path, 'logs.pkl')
    with open(logs_path, 'rb') as f:
        logs = pickle.load(f)

    # Gathers a list of pareto points
    pareto_keys = []
    pareto_pop = logs['pareto'][-1]['population']

    for gene in pareto_pop:
        # Converts gene to a string-based definition (primary key)
        key = converter.gene_to_key(gene)

        if not key in pareto_keys:
            pareto_keys.append(key)

    pop_configs = []

    is_pareto = []
    is_pareto_dict = {}

    seen = {}
    latencies = {}

    for i, pop in enumerate(logs['population']):
        for idx, gene in enumerate(pop):
            key = converter.gene_to_key(gene)

            if not key in seen.keys():
                seen[key] = 1

                # Converts gene to configuration and appends to population
                model_config = converter.gene_to_config(gene)
                pop_configs.append(model_config)

                if key in pareto_keys:
                    is_pareto.append(True)
                else:
                    is_pareto.append(False)

                if i < len(logs['latencies']):
                    latencies[key] = logs['latencies'][i][idx]
                    is_pareto_dict[key] = is_pareto[-1]

    print(f'Total jobs: {len(pop_configs)} | Pareto-frontier jobs: {np.sum(is_pareto)}')
    assert np.sum(list(is_pareto_dict.values())) == np.sum(is_pareto)

    latency_output_path = os.path.join(results_path, 'latencies.pkl')
    with open(latency_output_path, 'wb') as f:
        pickle.dump(latencies, f)

    pareto_output_path = os.path.join(results_path, 'pareto.pkl')
    with open(pareto_output_path, 'wb') as f:
        pickle.dump(is_pareto_dict, f)

    create_jobs(pop_configs,
                default_config_file=default_config_file,
                model_type=model_type,
                max_step=max_step,
                start_config=start_config,
                n_jobs=n_jobs,
                n_gpus=n_gpus,
                gpu_config=gpu_config, 
                is_pareto=is_pareto,
                output_path=output_path)
