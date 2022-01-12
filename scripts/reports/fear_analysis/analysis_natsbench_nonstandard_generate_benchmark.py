# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from typing import Dict, List, Type, Iterator, Tuple
import glob
import os
import pathlib
from collections import OrderedDict, defaultdict, namedtuple
from scipy.stats.stats import _two_sample_transform
import yaml
from inspect import getsourcefile
import seaborn as sns
import math as ma


import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from scipy.stats import kendalltau, spearmanr

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
    parser = argparse.ArgumentParser(description='Report creator')
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
    num_workers = 60
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

    # remove all arch_ids which did not finish
    for key in list(logs.keys()):
        to_delete = False

        if 'eval_arch' not in list(logs[key].keys()):
            to_delete = True

        if to_delete:
            print(f'arch id {key} did not finish. removing from calculations.')
            logs.pop(key)
            continue

        # eval_arch may not have finished
        num_epochs = confs[key]['nas']['eval']['trainer']['epochs']
        last_epoch_key = int(list(logs[key]['eval_arch']['eval_train']['epochs'].keys())[-1])
        if last_epoch_key != num_epochs - 1:
            print(f'arch id {key} did not finish. removing from calculations.')
            logs.pop(key)

    
    # create a dict with arch_id: regular eval score as entries
    # and save since synthetic cifar10 or other new datasets 
    # are not part of the benchmark
    arch_id_reg_eval = {}
    arch_id_params_flops = {}
    arch_id_trainacc_at_n_epoch = {}
    n_epoch = '3'

    for key in logs.keys():
        arch_id = confs[key]['nas']['eval']['natsbench']['arch_index']
        reg_eval = logs[key]['eval_arch']['eval_train']['best_test']['top1']
        train_acc_at_n = logs[key]['eval_arch']['eval_train']['epochs'][n_epoch]['train']['top1']
        num_params = logs[key]['eval_arch']['eval_train']['num_params']
        mega_flops_per_batch = logs[key]['eval_arch']['eval_train']['mega_flops_per_batch']
        # store
        arch_id_reg_eval[arch_id] = reg_eval
        arch_id_trainacc_at_n_epoch[arch_id] = train_acc_at_n
        arch_id_params_flops[arch_id] = {'params': num_params, 'flops': mega_flops_per_batch}

    savename = os.path.join(out_dir, 'arch_id_test_accuracy.yaml')
    with open(savename, 'w') as f:
        yaml.dump(arch_id_reg_eval, f)

    savename = os.path.join(out_dir, 'arch_id_params_flops.yaml')
    with open(savename, 'w') as f:
        yaml.dump(arch_id_params_flops, f)

    # now create a list of regular evaluation and corresponding synflow scores
    # to compute spearman's correlation
    all_reg_evals = []
    all_epochs_at_n = []
    for arch_id in arch_id_reg_eval.keys():
        all_reg_evals.append(arch_id_reg_eval[arch_id])
        all_epochs_at_n.append(arch_id_trainacc_at_n_epoch[arch_id])

    spe_epochs_at_n, _ = spearmanr(all_reg_evals, all_epochs_at_n)
    print(f'Spearman corr. {n_epoch}: {spe_epochs_at_n}')
        
    print(f'num valid architectures used for analysis {len(logs)}')

    # plot histogram of regular evaluation scores
    fig = px.histogram(all_reg_evals, labels={'x': 'Test Accuracy', 'y': 'Counts'})
    savename = os.path.join(out_dir, 'distribution_of_test_accuracies.html')
    fig.write_html(savename)
    #fig.show()

    # plot histogram of training scores
    all_train_accs = []
    for key in logs.keys():
        train_acc = logs[key]['eval_arch']['eval_train']['best_train']['top1']
        all_train_accs.append(train_acc)

    fig1 = px.histogram(all_train_accs, labels={'x': 'Train Accuracy', 'y': 'Counts'})
    savename = os.path.join(out_dir, 'distribution_of_train_accuracies.html')
    fig1.write_html(savename)
    #fig1.show()

    



if __name__ == '__main__':
    main()