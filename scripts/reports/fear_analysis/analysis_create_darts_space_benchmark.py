# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import argparse
from typing import Dict, List, Type, Iterator, Tuple
import glob
import os
import pathlib
from collections import OrderedDict, defaultdict
from scipy.stats.stats import _two_sample_transform
import yaml
from inspect import getsourcefile
import seaborn as sns
import math as ma


import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from scipy.stats import kendalltau, spearmanr, sem

from runstats import Statistics

#import matplotlib
#matplotlib.use('TkAgg')
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from collections import namedtuple


from archai.common import utils
from archai.common.ordereddict_logger import OrderedDictLogger
from archai.common.analysis_utils import epoch_nodes, parse_a_job, fix_yaml, remove_seed_part, group_multi_runs, collect_epoch_nodes, EpochStats, FoldStats, stat2str, get_epoch_stats, get_summary_text, get_details_text, plot_epochs, write_report

import re

def main():
    parser = argparse.ArgumentParser(description='Generates the darts space benchmark')
    parser.add_argument('--results-dir', '-d', type=str,
                        default=r'~/logdir/proxynas_test_0001',
                        help='folder with experiment results from pt')
    parser.add_argument('--out-dir', '-o', type=str, default=r'~/logdir/reports',
                        help='folder to output reports')
    args, extra_args = parser.parse_known_args()

    # root dir where all results are stored
    results_dir = pathlib.Path(utils.full_path(args.results_dir))
    print(f'results_dir: {results_dir}')

    # extract experiment name which is top level directory
    exp_name = results_dir.parts[-1]

    # create results dir for experiment
    out_dir = utils.full_path(os.path.join(args.out_dir, exp_name))
    print(f'out_dir: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

    # get list of all structured logs for each job
    logs = {}
    confs = {}
    job_dirs = list(results_dir.iterdir())

    # # test single job parsing for debugging
    # # WARNING: very slow, just use for debugging
    # for job_dir in job_dirs:
    #     a = parse_a_job(job_dir)

    # parallel parsing of yaml logs
    num_workers = 48
    with Pool(num_workers) as p:
        a = p.map(parse_a_job, job_dirs)

    for storage in a:
        for key, val in storage.items():
            logs[key] = val[0]
            confs[key] = val[1]

    # remove all search jobs
    for key in list(logs.keys()):
        if 'search' in key:
            logs.pop(key)

    # check for problematic logs
    for key in list(logs.keys()):
        if 'best_test' not in logs[key]['regular_evaluate']['eval_arch']['eval_train']:
            print(f'problem in {key}')
            logs.pop(key)

    archid_testacc = {}
    archid_params = {}
    archid_flops = {}
    for key in logs.keys():
        if 'eval' in key:
            try:
                dataset_name = confs[key]['dataset']['name']
                if dataset_name == 'darcyflow':
                    test_acc = -logs[key]['regular_evaluate']['eval_arch']['eval_train']['best_test']['loss']
                else:    
                    test_acc = logs[key]['regular_evaluate']['eval_arch']['eval_train']['best_test']['top1']
                arch_id = confs[key]['nas']['eval']['dartsspace']['arch_index']
                archid_testacc[arch_id] = test_acc
                
                # get the number of params if in logs (most have it unless the early part is missing)
                if 'num_params' in logs[key]['regular_evaluate']['eval_arch']['eval_train']:
                    num_params = logs[key]['regular_evaluate']['eval_arch']['eval_train']['num_params']
                    archid_params[arch_id] = num_params
                    mega_flops_per_batch = logs[key]['regular_evaluate']['eval_arch']['eval_train']['mega_flops_per_batch']
                    archid_flops[arch_id] = mega_flops_per_batch

            except KeyError as err:
                print(f'KeyError {err} not in {key}!')
                sys.exit()

    print(f'Number of archs in benchmark {len(archid_params)}')    

    # save accuracies
    savename = os.path.join(out_dir, 'arch_id_test_accuracy.yaml')
    with open(savename, 'w') as f:
        yaml.dump(archid_testacc, f)

    # save params flops
    arch_id_params_flops = dict()
    savename = os.path.join(out_dir, 'arch_id_params_flops.yaml')
    for archid in archid_params.keys():
        num_params = archid_params[archid]
        num_flops = archid_flops[archid]
        arch_id_params_flops[archid] = {'params': num_params, 'flops': num_flops}

    with open(savename, 'w') as f:
        yaml.dump(arch_id_params_flops, f)

    # plot test accuracy vs. number of params 
    # to see how the distribution looks
    testaccs = []
    params = []
    flops = []
    for archid in archid_params.keys():
        num_params = archid_params[archid]
        test_acc = archid_testacc[archid]
        num_flops = archid_flops[archid]
        testaccs.append(test_acc)
        params.append(num_params)
        flops.append(num_flops)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=testaccs, y=params, mode='markers'))
    fig.update_layout(xaxis_title="Test Accuracy", 
                    yaxis_title="Parameters")
    fig.update_layout(font=dict(size=36)) # font size
    fig.update_traces(marker=dict(size=20)) # marker size

    savename_html = os.path.join(out_dir, 'darts_space_params_vs_test_acc.html')
    fig.write_html(savename_html)

    savename_png = os.path.join(out_dir, 'darts_space_params_vs_test_acc.png')
    fig.write_image(savename_png, width=1500, height=1500, scale=1)

    # compute spearman correlation of #params vs. test accuracy
    param_spe, param_sp_value = spearmanr(testaccs, params)
    flop_spe, flop_sp_value = spearmanr(testaccs, flops)
    print(f'Spearman correlation of #params vs. test accuracy is {param_spe}')
    print(f'Spearman correlation of #flops vs. test accuracy is {flop_spe}')
    savename = os.path.join(out_dir, 'darts_space_params_flops_spe.txt')
    with open(savename, 'w') as f:
        f.write(f'Spe #params vs. test accuracy: {param_spe}')
        f.write(f'Spe #flops vs. test accuracy: {flop_spe}')
    





if __name__ == '__main__':
    main()