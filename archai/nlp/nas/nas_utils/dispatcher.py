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


def check_job_status(exp_name: str,
                     n_configs: int,
                     start_config: Optional[int] = 0) -> None:
    """Checks the status of a particular job.

    Args:
        exp_name: Name of the experiment.
        n_configs: Number of configurations to check.
        start_config: Starting range of the configuration to be checked.

    """

    pass_count = 0

    while pass_count < n_configs:
        print('Waiting for 1 minute before checking job status...')
        time.sleep(60)

        os.system(f'amlt status  {exp_name} > tmp.txt')
        with open('tmp.txt', 'r') as f:
            lines = f.readlines()

        pass_count = 0
        for i in range(len(lines)):
            l = lines[i].split()

            if len(l) == 0:
                continue

            if ':config_' in l[0]:
                config_idx = int(re.search(':config_([0-9]+)', l[0]).group(1))
                print(f'Checking status of job {config_idx}')

                if (config_idx < start_config) or (config_idx >= (start_config + n_configs)):
                    print('Supplied index is not in valid range')
                    continue

                if 'pass' in l:
                    pass_count += 1

                elif 'failed' in l:
                    assert False, f'Experiment {exp_name}, job :config_{config_idx} failed'

        print(f'{pass_count} total jobs finished.')

    os.system('rm tmp.txt')


def create_job_command(configs: List[Dict[str, Any]],
                       max_step: int,
                       n_gpus: int,
                       model_type: str,
                       gpu_config: str,
                       is_pareto: Optional[bool] = None) -> str:
    """Creates a command-line command that launches a new job.

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
                target: Optional[str] = 'NLX-NDv2',
                exp_name: Optional[str] = 'midevolution_training_',
                is_pareto: Optional[bool] = None,
                path_to_save: Optional[str] = None) -> Tuple[str, str, int]:
    """Creates a batch of jobs.

    Args:
        all_population: List of genes that belongs to the population.
        start_config: Starting range of the configuration to be checked.
        n_jobs: Number of jobs to be created.
        max_step: Number of maximum steps to train the models.
        n_gpus: Number of GPUs to be used.
        model_type: Type of model.
        gpu_config: GPU configuration.
        target: Target machine to deploy the jobs.
        exp_name: Name of the experiment.
        is_pareto: Whether job is from pareto front or not.
        path_to_save: Folder that output files should be saved on.

    Returns:
        (Tuple[str, str, int]): Experiment name, bash file with created commands and range
            of configurations that will be deployed.
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
            # NOTE: changed += to =  not clear what is right.
            amlt_config['jobs'][0]['command'] = create_job_command(copy.deepcopy(all_population[c:c+n_jobs]), max_step, n_gpus, model_type, gpu_config, is_pareto=is_pareto[c:c+n_jobs])
        else:
            amlt_config['jobs'][0]['command'] = create_job_command(copy.deepcopy(all_population[c:c+n_jobs]), max_step, n_gpus, model_type, gpu_config)

        config_file = 'nv_train_'+str(config_idx)+'.yaml'
        f_name = os.path.join(path_to_configs, config_file)
        with open(f_name, 'w') as f:
            yaml.dump(amlt_config, f)

        c += n_jobs
        config_idx += 1

    exp_name = exp_name + str(max_step)

    bash_f_name = 'amlt_run'
    bash_file = os.path.join(path_to_configs, bash_f_name+'.sh')

    if os.path.exists(bash_file):
        os.remove(bash_file)

    for i in range(start_config, config_idx):
        with open(bash_file, 'a') as f:
            f.write('amlt run --yes archai/nlp/nvidia_transformer_xl/configs/nv_train_{}.yaml {} -t {}\n'.format(i, exp_name, target))

    return exp_name, bash_file, config_idx-start_config


def submit_ground_truth_jobs(args: Dict[str, Any],
                             alg: Any,
                             max_step: Optional[int] = 500,
                             start_config: Optional[int] = 0,
                             n_jobs: Optional[int] = 50,
                             n_gpus: Optional[int] = 8,
                             model_type: Optional[str] = 'mem_transformer',
                             gpu_config: Optional[str] = 'dgx1_8gpu_fp32',
                             targets: Optional[List[str]] = ['NLX-NDv2']) -> None:
    """Submits a batch of ground-truth jobs.

    Args:
        args: Additional arguments.
        alg: Evolutionary algorithm.
        max_step: Number of maximum steps to train the models.
        start_config: Starting range of the configuration to be checked.
        n_jobs: Number of jobs to be created.
        n_gpus: Number of GPUs to be used.
        model_type: Type of model.
        gpu_config: GPU configuration.
        targets: Target machines to deploy the experiments.

    """

    # get amlt bash files for running all jobs to get the ground-truth Pareto
    results_path = args['results_path']
    path_to_logs = os.path.join(results_path, 'logs.pkl')

    with open(path_to_logs, 'rb') as f:
        logs = pickle.load(f)

    # get a list of pareto points
    pareto_keys = []
    pareto_pop = logs['pareto'][-1]['population']

    for gene in pareto_pop:
        key = alg.converter.gene_to_str(gene)  # ','.join([str(g) for g in gene])
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
            key = alg.converter.gene_to_str(gene)

            if not key in seen.keys():
                seen[key] = 1
                model_config = alg.converter.gene_to_config(gene)
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
                model_type, gpu_config, targets[0], exp_name='evolution_', is_pareto=is_pareto)


def submit_pareto_front_jobs(args: Dict[str, Any],
                             alg: Any,
                             is_pareto_dict: Dict[str, Any],
                             max_step: Optional[int] = 500,
                             start_config: Optional[int] = 0,
                             n_jobs: Optional[int] = 50,
                             n_gpus: Optional[int] = 8,
                             gpu_config: Optional[str] = 'dgx1_8gpu_fp32',
                             targets: Optional[List[str]] = ['NLX-NDv2']) -> None:
    """Submits a batch of Pareto front jobs.

    Args:
        args: Additional arguments.
        alg: Evolutionary algorithm.
        is_pareto_dict: Pareto front configurations.
        max_step: Number of maximum steps to train the models.
        start_config: Starting range of the configuration to be checked.
        n_jobs: Number of jobs to be created.
        n_gpus: Number of GPUs to be used.
        gpu_config: GPU configuration.
        targets: Target machines to deploy the experiments.

    """
    
    # get amlt bash files for training jobs of selected Pareto points
    results_path = args['results_path']
    path_to_logs = os.path.join(results_path, 'logs.pkl')

    with open(path_to_logs, 'rb') as f:
        logs = pickle.load(f)

    seen = {}
    pareto_population = []
    pareto_latencies = {}

    for iter, pop in enumerate(logs['population']):
        for idx, gene in enumerate(pop):
            key = alg.converter.gene_to_str(gene)

            if not key in seen.keys():
                seen[key] = 1
                model_config = alg.converter.gene_to_config(gene)
                if is_pareto_dict[key]:
                    pareto_population.append(model_config)

                if iter < len(logs['latencies']):
                    pareto_latencies[key] = logs['latencies'][iter][idx]

    print('{} total configs and {} on the pareto'.format(len(seen.keys()), np.sum(len(pareto_population))))
    assert np.sum(list(is_pareto_dict.values())) == len(pareto_population)

    path_to_pkl = os.path.join(results_path, 'pareto_latencies.pkl')
    with open(path_to_pkl, 'wb') as f:
        pickle.dump(pareto_latencies, f)

    create_jobs(pareto_population, start_config, n_jobs, max_step, n_gpus, gpu_config, targets[0],
                exp_name='evolution_', path_to_save=os.path.join('../archaiphilly/transformer_nas', 'pareto_'+args['device_name']))
