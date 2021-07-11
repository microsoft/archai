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
    parser = argparse.ArgumentParser(description='Freeze Darts Space Experiments')
    parser.add_argument('--results-dir', '-d', type=str,
                        default=r'~/logdir/proxynas_test_0001',
                        help='folder with experiment results')
    parser.add_argument('--out-dir', '-o', type=str, default=r'~/logdir/reports',
                        help='folder to output reports')
    parser.add_argument('--reg-evals-file', '-r', type=str,
                        help='yaml file which contains full evaluation results for every archid')
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

    # regular full training file
    with open(args.reg_evals_file, 'r') as f:
        reg_evals_data = yaml.load(f, Loader=yaml.Loader)

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

    

    all_reg_evals = []
    all_freeze_evals_last = []

    all_freeze_time_last = []
    all_cond_time_last = []
    all_partial_time_last = []

    num_archs_unmet_cond = 0

    for key in logs.keys():
        if 'eval' in key:
            try:
                # if at the end of conditional training train accuracy has not gone above target then don't consider it
                # important to get this first
                last_cond_epoch_key = list(logs[key]['freeze_evaluate']['eval_arch']['conditional_training']['eval_train']['epochs'].keys())[-1]
                use_val = confs[key]['nas']['eval']['trainer']['use_val']
                threshold = confs[key]['nas']['eval']['trainer']['top1_acc_threshold']
                if use_val:
                    val_or_train = 'val'
                else:
                    val_or_train = 'train'
                end_cond = logs[key]['freeze_evaluate']['eval_arch']['conditional_training']['eval_train']['epochs'][last_cond_epoch_key][val_or_train]['top1']
                if end_cond < threshold:
                    num_archs_unmet_cond += 1
                    continue

                # regular evaluation
                # important to get this first since if an arch id is
                # not in the benchmark we need to remove it from consideration
                arch_id = confs[key]['nas']['eval']['dartsspace']['arch_index']
                if arch_id not in list(reg_evals_data.keys()):
                    continue
                reg_eval_top1 = reg_evals_data[arch_id]
                all_reg_evals.append(reg_eval_top1)

                # freeze evaluation 
                #--------------------

                # at last epoch
                last_freeze_epoch_key = list(logs[key]['freeze_evaluate']['eval_arch']['freeze_training']['eval_train']['epochs'].keys())[-1]
                freeze_eval_top1 = logs[key]['freeze_evaluate']['eval_arch']['freeze_training']['eval_train']['epochs'][last_freeze_epoch_key][val_or_train]['top1']                                            
                all_freeze_evals_last.append(freeze_eval_top1)

                # collect duration for conditional training and freeze training
                # NOTE: don't use val_or_train here since we are really interested in the duration of training 
                freeze_duration = 0.0
                for epoch_key in logs[key]['freeze_evaluate']['eval_arch']['freeze_training']['eval_train']['epochs']:
                    freeze_duration += logs[key]['freeze_evaluate']['eval_arch']['freeze_training']['eval_train']['epochs'][epoch_key]['train']['duration']

                cond_duration = 0.0
                for epoch_key in logs[key]['freeze_evaluate']['eval_arch']['conditional_training']['eval_train']['epochs']:
                    cond_duration += logs[key]['freeze_evaluate']['eval_arch']['conditional_training']['eval_train']['epochs'][epoch_key]['train']['duration']

                all_freeze_time_last.append(freeze_duration + cond_duration)
                all_cond_time_last.append(cond_duration)
                all_partial_time_last.append(freeze_duration)

            except KeyError as err:
                print(f'KeyError {err} not in {key}!')
                sys.exit()


    # Store some key numbers in results.txt
    results_savename = os.path.join(out_dir, 'results.txt')
    with open(results_savename, 'w') as f:
        f.write(f'Number of archs which did not reach condition: {num_archs_unmet_cond} \n')
        f.write(f'Total valid archs processed: {len(all_reg_evals)} \n')

    print(f'Number of archs which did not reach condition: {num_archs_unmet_cond}')
    print(f'Total valid archs processed: {len(all_reg_evals)}')

    # Sanity check
    assert len(all_reg_evals) == len(all_freeze_evals_last)
    assert len(all_reg_evals) == len(all_cond_time_last)
    assert len(all_reg_evals) == len(all_freeze_time_last)

    # scatter plot between time to threshold accuracy and regular evaluation
    fig = px.scatter(x=all_cond_time_last, y=all_reg_evals, labels={'x': 'Time to reach threshold train accuracy (s)', 'y': 'Final Accuracy'})
    fig.update_layout(font=dict(
        size=48,
    ))

    savename = os.path.join(out_dir, 'cond_time_vs_final_acc.html')
    fig.write_html(savename)
    savename_pdf = os.path.join(out_dir, 'cond_time_vs_final_acc.pdf')
    fig.write_image(savename_pdf, engine="kaleido", width=1500, height=1500, scale=1)
    fig.show()

    # histogram of training accuracies
    fig = px.histogram(all_reg_evals, labels={'x': 'Test Accuracy', 'y': 'Counts'})
    savename = os.path.join(out_dir, 'distribution_of_reg_evals.html')
    fig.write_html(savename)
    fig.show()

    # Freeze training results at last epoch        
    freeze_tau, freeze_p_value = kendalltau(all_reg_evals, all_freeze_evals_last)
    freeze_spe, freeze_sp_value = spearmanr(all_reg_evals, all_freeze_evals_last)
    print(f'Freeze Kendall Tau score: {freeze_tau:3.03f}, p_value {freeze_p_value:3.03f}')
    print(f'Freeze Spearman corr: {freeze_spe:3.03f}, p_value {freeze_sp_value:3.03f}')
    with open(results_savename, 'a') as f:
        f.write(f'Freeze Kendall Tau score: {freeze_tau:3.03f}, p_value {freeze_p_value:3.03f} \n')
        f.write(f'Freeze Spearman corr: {freeze_spe:3.03f}, p_value {freeze_sp_value:3.03f} \n')

    plt.clf()
    sns.scatterplot(x=all_reg_evals, y=all_freeze_evals_last)
    plt.xlabel('Test top1 at natsbench full training')
    plt.ylabel('Freeze training')
    plt.grid()
    savename = os.path.join(out_dir, 'proxynas_freeze_training_epochs.png')
    plt.savefig(savename, dpi=plt.gcf().dpi, bbox_inches='tight')


if __name__ == '__main__':
    main()