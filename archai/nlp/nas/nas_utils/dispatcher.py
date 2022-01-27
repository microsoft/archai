# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Auxiliary methods that dispatches traning/evaluation jobs.
"""

import copy
import os
import pickle
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from archai.nlp.nas.nas_utils.parser import parse_values_from_yaml
from archai.nlp.nas.nas_utils.converter import Converter


def create_job_wt103_command(configs: List[Dict[str, Any]],
                            max_step: int,
                            n_gpus: int,
                            model_type: str,
                            gpu_config: str,
                            is_pareto: Optional[bool] = None) -> str:
    """Creates a command-line command that launches a new job.
    WARNING: specific to WT103 dataset.

    Args:
        configs: List of configuration dictionaries.
        max_step: Maximum number of training steps.
        n_gpus: Number of GPUs.
        model_type: Type of model.
        gpu_config: GPU configuration to be used.
        is_pareto: Whether job is from pareto front or not.

    Returns:
        (str): Command-line command.

    """

    command = []

    for i, config in enumerate(configs):
        # WARNING: div_val is 4 hardcoded here!
        config['div_val'] = 4
        config['d_embed'] = config['d_model']
        config['d_head'] = [config['d_model'] // n_head for n_head in config['n_head']]

        for k, v in config.items():
            config[k] = str(parse_values_from_yaml(v))

        exp_name = 'j' + str(i) + ('_pareto' if (is_pareto is not None and is_pareto[i]) else '')

        command.append('python -m torch.distributed.launch --nproc_per_node="%s" archai/nlp/train.py --model_type %s --config %s \
                       --config_file wt103_base.yaml --n_layer %s --n_head %s --d_model %s --d_head %s \
                       --d_inner %s --d_embed %s --div_val %s --max_step %d --experiment_name %s'
                       % (str(n_gpus), model_type, gpu_config, config['n_layer'], config['n_head'], config['d_model'], config['d_head'], config['d_inner'],
                       config['d_embed'], config['div_val'], max_step, exp_name))

    return command


def create_jobs(all_population: List[List[Any]],
                start_config: Optional[int] = 0,
                n_jobs: Optional[int] = 50,
                max_step: Optional[int] = 500,
                n_gpus: Optional[int] = 8,
                model_type: Optional[str] = 'mem_transformer',
                gpu_config: Optional[str] = 'dgx1_8gpu_fp32',
                is_pareto: Optional[bool] = None,
                path_to_save: Optional[str] = None) -> None:
    """Creates a batch of jobs.

    Args:
        all_population: List of genes that belongs to the population.
        start_config: Starting range of the configuration to be checked.
        n_jobs: Number of jobs to be created.
        max_step: Number of maximum steps to train the models.
        n_gpus: Number of GPUs to be used.
        model_type: Type of model.
        gpu_config: GPU configuration.
        is_pareto: Whether job is from pareto front or not.
        path_to_save: Folder that output files should be saved on.

    Returns:
        None
    """

    path_to_configs = os.path.expanduser('~/configs') if not path_to_save else path_to_save
    os.makedirs(path_to_configs, exist_ok=True)

    # create corresponding yaml files for amulet jobs
    n_configs = len(all_population)
    c = 0
    config_idx = start_config

    while c < n_configs:
        amlt_config = {}
        amlt_config['jobs'] = [{}]

        if is_pareto is not None:
            amlt_config['jobs'][0]['command'] = create_job_wt103_command(copy.deepcopy(all_population[c:c+n_jobs]), max_step, n_gpus, model_type, gpu_config, is_pareto=is_pareto[c:c+n_jobs])
        else:
            amlt_config['jobs'][0]['command'] = create_job_wt103_command(copy.deepcopy(all_population[c:c+n_jobs]), max_step, n_gpus, model_type, gpu_config)

        config_file = 'nv_train_'+str(config_idx)+'.yaml'
        f_name = os.path.join(path_to_configs, config_file)
        with open(f_name, 'w') as f:
            yaml.dump(amlt_config, f)

        c += n_jobs
        config_idx += 1


def prepare_pareto_jobs(results_path:str, 
                        converter:Converter,
                        path_to_save:str):
    ''' Prepares command line for training pareto frontier jobs only for full training. '''

    path_to_logs = os.path.join(results_path, 'logs.pkl')
    
    with open(path_to_logs, 'rb') as f:
        logs = pickle.load(f)

    # get the list of pareto points
    pareto_pop = logs['pareto'][-1]['population']

    seen = set()

    configs_to_launch = []
    is_pareto = []

    for gene in pareto_pop:
        model_key = converter.gene_to_str(gene)
        model_config = converter.gene_to_config(gene)
        if model_key in seen:
            continue
        else:
            seen.add(model_key)
            configs_to_launch.append(model_config)
            is_pareto.append(True)


    create_jobs(configs_to_launch, start_config=0, n_jobs=10, max_step=40000,
    n_gpus=8, model_type='mem_transformer', gpu_config='dgx1_8gpu_fp32', 
    is_pareto=is_pareto, path_to_save=path_to_save)

        

def prepare_ground_truth_jobs(results_path: str,
                             converter: Converter,
                             max_step: Optional[int] = 500,
                             start_config: Optional[int] = 0,
                             n_jobs: Optional[int] = 50,
                             n_gpus: Optional[int] = 8,
                             model_type: Optional[str] = 'mem_transformer',
                             gpu_config: Optional[str] = 'dgx1_8gpu_fp32',                             
                             path_to_save: Optional[str] = './configs') -> None:
    """Prepares command lines for training all visited points during search for full training

    Args:
        results_path: Path to search results.
        converter: Converter class object.
        max_step: Number of maximum steps to train the models.
        start_config: Starting range of the configuration to be checked.
        n_jobs: Number of jobs to be created.
        n_gpus: Number of GPUs to be used.
        model_type: Type of model.
        gpu_config: GPU configuration.
        path_to_save: Save folder for the created command lines.
    """

    # get bash files for running all jobs to get the ground-truth Pareto
    path_to_logs = os.path.join(results_path, 'logs.pkl')

    with open(path_to_logs, 'rb') as f:
        logs = pickle.load(f)

    # get a list of pareto points
    pareto_keys = []
    pareto_pop = logs['pareto'][-1]['population']

    for gene in pareto_pop:
        key = converter.gene_to_str(gene)  # ','.join([str(g) for g in gene])
        if not key in pareto_keys:
            pareto_keys.append(key)

    print('number of paretos:', len(pareto_keys))
    print(len(logs['population']))

    seen = {}
    all_population = []
    is_pareto = []
    is_pareto_dict = {}
    all_latencies = {}

    for iter, pop in enumerate(logs['population']):
        unseen = 0

        for idx, gene in enumerate(pop):
            key = converter.gene_to_str(gene)

            if not key in seen.keys():
                seen[key] = 1
                model_config = converter.gene_to_config(gene)
                all_population.append(model_config)
                unseen += 1

                if key in pareto_keys:
                    is_pareto.append(True)
                else:
                    is_pareto.append(False)

                if iter < len(logs['latencies']):
                    all_latencies[key] = logs['latencies'][iter][idx]
                    is_pareto_dict[key] = is_pareto[-1]

    print('{} total configs and {} on the pareto'.format(len(all_population), np.sum(is_pareto)))
    assert np.sum(list(is_pareto_dict.values())) == np.sum(is_pareto)

    path_to_pkl = os.path.join(results_path, 'latencies.pkl')
    with open(path_to_pkl, 'wb') as f:
        pickle.dump(all_latencies, f)

    path_to_pkl = os.path.join(results_path, 'pareto.pkl')
    with open(path_to_pkl, 'wb') as f:
        pickle.dump(is_pareto_dict, f)

    create_jobs(all_population, start_config, n_jobs, max_step, n_gpus,
                model_type, gpu_config,
                is_pareto=is_pareto, path_to_save=path_to_save)


