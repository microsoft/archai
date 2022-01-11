# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Auxiliary tool that dispatches new jobs when conducting searches.
"""

import copy
import os
import pickle
import re
import time

import numpy as np
import yaml
from archai.nlp.nas.constraint_getter import get_yaml_values


model_config_defaults = {'d_head': None,
                         'n_token': 267736,
                         'dropout': 0.1,
                         'dropatt': 0.0,
                         'd_embed': None,
                         'div_val': 4,
                         'pre_lnorm': False,
                         'tgt_len': 192,
                         'ext_len': 0,
                         'mem_len': 192,
                         'same_length': False,
                         'attn_type': 0,
                         'clamp_len': -1,
                         'sample_softmax': -1,
                         'cutoffs': [19997, 39997, 199997],
                         'tie_projs': [False, True, True, True],
                         'tie_weight': True,
                         'dtype': None,
                         'primer_conv': False,
                         'primer_square': False,
                         'use_cache': False}


def check_job_status(exp_name, n_configs, start_config=0):
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
                print(f'checking status of job {config_idx}')

                if (config_idx < start_config) or (config_idx >= (start_config + n_configs)):
                    print('This job index is not in range')
                    continue

                if 'pass' in l:
                    pass_count += 1

                elif 'failed' in l:
                    assert False, f'experiment {exp_name}, job :config_{config_idx} failed'

        print(f'{pass_count} total amlt jobs finished so far.')

    os.system('rm tmp.txt')


def get_bundle_run_command(configs, max_step, n_gpus, gpu_config, is_pareto=None):
    command = []
    for i, curr_config in enumerate(configs):
        curr_config['d_embed'] = curr_config['d_model']
        curr_config['d_head'] = [curr_config['d_model'] // n_head for n_head in curr_config['n_head']]

        for k, v in curr_config.items():
            curr_config[k] = str(get_yaml_values(v))

        exp_name = 'j' + str(i) + ('_pareto' if (is_pareto is not None and is_pareto[i]) else '')

        command.append('python -m torch.distributed.launch --nproc_per_node="%s" archai/nlp/nvidia_transformer_xl/train.py --config %s \
                       --config_file wt103_base.yaml --n_layer %s --n_head %s --d_model %s --d_head %s \
                       --d_inner %s --d_embed %s --div_val %s --max_step %d --experiment_name %s'
                       % (str(n_gpus), gpu_config, curr_config['n_layer'], curr_config['n_head'], curr_config['d_model'], curr_config['d_head'], curr_config['d_inner'],
                       curr_config['d_embed'], model_config_defaults['div_val'], max_step, exp_name))

    return command


def create_jobs(all_population, start_config=0, bundle_count=50, max_step=500, n_gpus=8, gpu_config='dgx1_8gpu_fp32', target='NLX-NDv2',
                exp_name='midevolution_training_', is_pareto=None, path_to_save=None):
    path_to_configs = os.path.join('./archai/nlp/nvidia_transformer_xl', 'configs') if path_to_save is None else path_to_save
    os.makedirs(path_to_configs, exist_ok=True)

    # create corresponding yaml files for amulet jobs
    n_configs = len(all_population)
    c = 0
    config_idx = start_config

    while c < n_configs:
        with open('../archaiphilly/nv_train.yaml') as file:
            amlt_config = yaml.safe_load(file)

        amlt_config['environment']['setup'] = ['set -e -o xtrace', 'pip install --user tensorboard']

        if target == 'NLX-NDV2':
            amlt_config['environment']['image'] = 'mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.0.3-cudnn8-ubuntu18.04:latest'
            amlt_config['environment']['registry'] = 'mcr.microsoft.com'
        else:
            amlt_config['environment']['image'] = 'debadeepta/pytorch:1.7.0-cuda11.0-cudnn8-devel'

        del amlt_config['search']

        amlt_config['jobs'] = [{}]
        amlt_config['jobs'][0]['name'] = 'config_{}'.format(str(config_idx))
        amlt_config['jobs'][0]['sku'] = f'G{n_gpus}'
        amlt_config['jobs'][0]['command'] = ['set -e -o xtrace', 'pip install --user -e .']

        if is_pareto is not None:
            amlt_config['jobs'][0]['command'] += get_bundle_run_command(copy.deepcopy(all_population[c:c+bundle_count]), max_step, n_gpus, gpu_config, is_pareto=is_pareto[c:c+bundle_count])
        else:
            amlt_config['jobs'][0]['command'] += get_bundle_run_command(copy.deepcopy(all_population[c:c+bundle_count]), max_step, n_gpus, gpu_config)

        config_file = 'nv_train_'+str(config_idx)+'.yaml'

        f_name = os.path.join(path_to_configs, config_file)
        with open(f_name, 'w') as file:
            yaml.dump(amlt_config, file)

        c += bundle_count
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


def submit_gt_jobs(args, alg, max_step=500, start_config=0, bundle_count=50, n_gpus=8, gpu_config='dgx1_8gpu_fp32', targets=['NLX-NDv2']):
    # get amlt bash files for running all jobs to get the ground-truth Pareto
    results_path = args['results_path']
    path_to_logs = os.path.join(results_path, 'logs.pkl')

    with open(path_to_logs, 'rb') as f:
        logs = pickle.load(f)

    # get a list of pareto points
    pareto_keys = []
    pareto_pop = logs['pareto'][-1]['population']

    for gene in pareto_pop:
        key = alg.converter.gene2key(gene)  # ','.join([str(g) for g in gene])
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
            key = alg.converter.gene2key(gene)

            if not key in seen.keys():
                seen[key] = 1
                model_config = alg.converter.gene2config(gene)
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

    create_jobs(all_population, start_config, bundle_count, max_step, n_gpus,
                gpu_config, targets[0], exp_name='evolution_', is_pareto=is_pareto)


def submit_pareto_front_jobs(args, alg, is_pareto_dict, max_step=500, start_config=0, bundle_count=50, n_gpus=8, gpu_config='dgx1_8gpu_fp32', targets=['NLX-NDv2']):
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
            key = alg.converter.gene2key(gene)

            if not key in seen.keys():
                seen[key] = 1
                model_config = alg.converter.gene2config(gene)
                if is_pareto_dict[key]:
                    pareto_population.append(model_config)

                if iter < len(logs['latencies']):
                    pareto_latencies[key] = logs['latencies'][iter][idx]

    print('{} total configs and {} on the pareto'.format(len(seen.keys()), np.sum(len(pareto_population))))
    assert np.sum(list(is_pareto_dict.values())) == len(pareto_population)

    path_to_pkl = os.path.join(results_path, 'pareto_latencies.pkl')
    with open(path_to_pkl, 'wb') as f:
        pickle.dump(pareto_latencies, f)

    create_jobs(pareto_population, start_config, bundle_count, max_step, n_gpus, gpu_config, targets[0],
                exp_name='evolution_', path_to_save=os.path.join('../archaiphilly/transformer_nas', 'pareto_'+args['device_name']))
