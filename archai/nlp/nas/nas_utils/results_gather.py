# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Gathers results and produces more comprehensive insights.
"""

import re
import time

import numpy as np

from archai.nlp.nas.nas_utils.parser import parse_results_from_experiment


def gather_amulet_results(population_size, exp_name, path_to_results, bundle_count, n_configs, start_config):
    keys = []

    for i in range(start_config, start_config + n_configs):
        for j in range(bundle_count):
            if len(keys) == population_size:
                break
            keys.append(f'config_{i}_j{j}')

    print(keys)

    def found_all_jobs(keys, results):
        for k in keys:
            if k not in results.keys():
                return False
        return True

    results = parse_results_from_experiment(exp_name, path_to_results, filetypes='.json')

    while not found_all_jobs(keys, results):
        print(population_size)
        time.sleep(60)
        results = parse_results_from_experiment(exp_name, path_to_results, filetypes='.json')

    configs = parse_results_from_experiment(exp_name, path_to_results, filetypes='.yaml')

    results_this_experiment = {k: results[k] for k in keys}
    configs_from_jobs = {k: {'d_model': configs[k]['d_model'], 'n_layer': configs[k]['n_layer'],
                             'd_inner': configs[k]['d_inner'], 'n_head': configs[k]['n_head']} for k in keys}

    configs_list = []
    val_ppls = np.zeros(population_size)
    indices = []

    for k, v in results_this_experiment.items():
        config_num = int(re.search('config_([0-9]+)', k).group(1))
        job_num = int(re.search('j([0-9]+)', k).group(1))

        val_ppls[(config_num * bundle_count) + job_num] = v['valid_perplexity']

        configs_list.append(configs_from_jobs[k])
        indices.append((config_num * bundle_count) + job_num)

    configs_list_sorted = []
    for i in range(len(configs_list)):
        idx = indices.index(i)
        configs_list_sorted.append(configs_list[idx])

    return val_ppls, configs_list_sorted
