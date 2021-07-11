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
    num_workers = 9
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
    for key in logs.keys():
        if 'best_test' not in logs[key]['regular_evaluate']['eval_arch']['eval_train']:
            print(f'problem in {key}')

    archid_testacc = {}
    archid_params = {}
    for key in logs.keys():
        if 'eval' in key:
            try:
                test_acc = logs[key]['regular_evaluate']['eval_arch']['eval_train']['best_test']['top1']
                arch_id = confs[key]['nas']['eval']['dartsspace']['arch_index']
                print(f'arch_index {arch_id}, test_acc {test_acc}')
                archid_testacc[arch_id] = test_acc

                # get the number of params if in logs (most have it unless the early part is missing)
                if 'num_params' in logs[key]['regular_evaluate']['eval_arch']['eval_train']:
                    num_params = logs[key]['regular_evaluate']['eval_arch']['eval_train']['num_params']
                    archid_params[arch_id] = num_params

            except KeyError as err:
                print(f'KeyError {err} not in {key}!')
                sys.exit()    

    savename = os.path.join(out_dir, 'darts_benchmark.yaml')
    with open(savename, 'w') as f:
        yaml.dump(archid_testacc, f)

    # plot test accuracy vs. number of params 
    # to see how the distribution looks
    testaccs = []
    params = []
    for archid in archid_params.keys():
        num_params = archid_params[archid]
        test_acc = archid_testacc[archid]
        testaccs.append(test_acc)
        params.append(num_params)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=params, y=testaccs, mode='markers'))
    fig.update_layout(title_text="Test accuracy vs. number of params on Darts Space Samples",
                    xaxis_title="Params", 
                    yaxis_title="Test Accuracy")

    savename_html = os.path.join(out_dir, 'darts_space_params_vs_test_acc.html')
    fig.write_html(savename_html)
    fig.show()

    # compute spearman correlation of #params vs. test accuracy
    param_spe, param_sp_value = spearmanr(testaccs, params)
    print(f'Spearman correlation of #params vs. test accuracy is {param_spe}')

    





if __name__ == '__main__':
    main()