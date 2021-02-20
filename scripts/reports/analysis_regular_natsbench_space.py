# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from typing import Dict, List, Type, Iterator, Tuple
import glob
import os
import pathlib
from collections import OrderedDict, defaultdict
from scipy.stats.stats import _two_sample_transform
import yaml
from inspect import getsourcefile
import re
from tqdm import tqdm
import seaborn as sns
import math as ma

from scipy.stats import kendalltau, spearmanr

from runstats import Statistics

#import matplotlib
#matplotlib.use('TkAgg')
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
from multiprocessing import Pool

from archai.common import utils
from archai.common.ordereddict_logger import OrderedDictLogger
from analysis_utils import epoch_nodes, parse_a_job, fix_yaml, remove_seed_part, group_multi_runs, collect_epoch_nodes, EpochStats, FoldStats, stat2str, get_epoch_stats, get_summary_text, get_details_text, plot_epochs, write_report

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
    num_workers = 8
    with Pool(num_workers) as p:
        a = p.map(parse_a_job, job_dirs)

    for storage in a:
        for key, val in storage.items():
            logs[key] = val[0]
            confs[key] = val[1]
                   
    # examples of accessing logs
    # best_test = logs[key]['eval_arch']['eval_train']['best_test']['top1']
    # best_train = logs[key]['eval_arch']['eval_train']['best_train']['top1']

    # remove all search jobs
    for key in list(logs.keys()):
        if 'search' in key:
            logs.pop(key)

    # sometimes a job has not even written logs to yaml
    for key in list(logs.keys()):
        if not logs[key]:
            print(f'arch id {key} did not finish. removing from calculations.')
            logs.pop(key)

    # remove all arch_ids which did not finish
    for key in list(logs.keys()):
        to_delete = False

        # it might have died early
        if 'eval_arch' not in list(logs[key].keys()):
            to_delete = True
 
        if 'regular_evaluate' not in list(logs[key].keys()):
            to_delete = True
        
        if to_delete:
            print(f'arch id {key} did not finish. removing from calculations.')
            logs.pop(key)
            continue

        if 'eval_train' not in list(logs[key]['eval_arch'].keys()):
            print(f'arch id {key} did not finish. removing from calculations.')
            logs.pop(key)
            continue

        if 'best_train' not in list(logs[key]['eval_arch']['eval_train'].keys()):
            print(f'arch id {key} did not finish. removing from calculations.')
            logs.pop(key)
            continue


    all_arch_ids = []
    all_reg_evals = []
    all_short_reg_evals = []
    all_short_reg_time = []
    

    for key in logs.keys():
        if 'eval' in key:
            try:                
                best_train = logs[key]['eval_arch']['eval_train']['best_train']['top1']
                all_short_reg_evals.append(best_train)
            
                # collect duration
                duration = 0.0
                for epoch_key in logs[key]['eval_arch']['eval_train']['epochs']:
                    duration += logs[key]['eval_arch']['eval_train']['epochs'][epoch_key]['train']['duration']

                all_short_reg_time.append(duration)
                
                # regular evaluation
                # --------------------
                reg_eval_top1 = logs[key]['regular_evaluate']['regtrainingtop1']
                all_reg_evals.append(reg_eval_top1)

                # record the arch id
                # --------------------
                if 'natsbench' in list(confs[key]['nas']['eval'].keys()):
                    all_arch_ids.append(confs[key]['nas']['eval']['natsbench']['arch_index'])
                elif 'nasbench101' in list(confs[key]['nas']['eval'].keys()):
                    all_arch_ids.append(confs[key]['nas']['eval']['nasbench101']['arch_index'])
                                
            except KeyError as err:
                print(f'KeyError {err} not in {key}!')


    # Store some key numbers in results.txt
    results_savename = os.path.join(out_dir, 'results.txt')
   
    # Sanity check
    assert len(all_reg_evals) == len(all_short_reg_evals)
    assert len(all_reg_evals) == len(all_short_reg_time)
    
    # Shortened training results       
    short_reg_tau, short_reg_p_value = kendalltau(all_reg_evals, all_short_reg_evals)
    short_reg_spe, short_reg_sp_value = spearmanr(all_reg_evals, all_short_reg_evals)
    print(f'Short reg Kendall Tau score: {short_reg_tau:3.03f}, p_value {short_reg_p_value:3.03f}')
    print(f'Short reg Spearman corr: {short_reg_spe:3.03f}, p_value {short_reg_sp_value:3.03f}')
    print(f'Valid archs: {len(all_reg_evals)}')
    with open(results_savename, 'w') as f:
        f.write(f'Short reg Kendall Tau score: {short_reg_tau:3.03f}, p_value {short_reg_p_value:3.03f} \n')
        f.write(f'Short reg Spearman corr: {short_reg_spe:3.03f}, p_value {short_reg_sp_value:3.03f} \n')

    plt.clf()
    sns.scatterplot(x=all_reg_evals, y=all_short_reg_evals)
    plt.xlabel('Test top1 at natsbench full training')
    plt.ylabel('Regular training with less epochs')
    plt.grid()
    savename = os.path.join(out_dir, 'shortened_training.png')
    plt.savefig(savename, dpi=plt.gcf().dpi, bbox_inches='tight')

    # Rank correlations at top n percent of architectures
    reg_shortreg_evals = [(all_reg_evals[i], all_short_reg_evals[i], all_short_reg_time[i]) for i in range(len(all_reg_evals))]

    # sort in descending order of accuracy of regular evaluation
    reg_shortreg_evals.sort(key=lambda x: x[0], reverse=True)

    top_percent_shortreg_times_avg = []
    top_percent_shortreg_times_std = []

    spe_shortreg_top_percents = []
    top_percents = []
    top_percent_range = range(2, 101, 2) 
    for top_percent in top_percent_range:
        top_percents.append(top_percent)
        num_to_keep = int(ma.floor(len(reg_shortreg_evals) * top_percent * 0.01))
        top_percent_evals = reg_shortreg_evals[:num_to_keep]
        top_percent_reg = [x[0] for x in top_percent_evals] 
        top_percent_shortreg = [x[1] for x in top_percent_evals]
        top_percent_shortreg_times = [x[2] for x in top_percent_evals]

        top_percent_shortreg_times_avg.append(np.mean(np.array(top_percent_shortreg_times)))
        top_percent_shortreg_times_std.append(np.std(np.array(top_percent_shortreg_times)))    

        spe_shortreg, _ = spearmanr(top_percent_reg, top_percent_shortreg)
        spe_shortreg_top_percents.append(spe_shortreg)
                
    plt.clf()
    sns.scatterplot(top_percents, spe_shortreg_top_percents)
    plt.legend(labels=['Shortened Regular Training'])
    plt.ylim((-1.0, 1.0))
    plt.xlim((0,100))
    plt.xlabel('Top percent of architectures')
    plt.ylabel('Spearman Correlation')
    plt.grid()
    savename = os.path.join(out_dir, f'spe_top_archs.png')
    plt.savefig(savename, dpi=plt.gcf().dpi, bbox_inches='tight')

    plt.clf()
    plt.errorbar(top_percents, top_percent_shortreg_times_avg, yerr=np.array(top_percent_shortreg_times_std)/2, marker='s', mfc='red', ms=10, mew=4)
    plt.xlabel('Top percent of architectures')
    plt.ylabel('Avg. time (s)')
    plt.yticks(np.arange(0,600, step=50))
    plt.grid()
    savename = os.path.join(out_dir, f'shortreg_train_duration_top_archs.png')
    plt.savefig(savename, dpi=plt.gcf().dpi, bbox_inches='tight')

    # time taken
    avg_shortreg_runtime = np.mean(np.array(all_short_reg_time))
    stderr_shortreg_runtime = np.std(np.array(all_short_reg_time)) / np.sqrt(len(all_short_reg_time))

    with open(results_savename, 'a') as f:
        f.write(f'Avg. Shortened Training Runtime: {avg_shortreg_runtime:.03f}, stderr {stderr_shortreg_runtime:.03f} \n')

    # how much overlap in top x% of architectures between method and groundtruth
    # ----------------------------------------------------------------------------
    arch_id_reg_evals =  [(arch_id, reg_eval) for arch_id, reg_eval in zip(all_arch_ids, all_reg_evals)]
    arch_id_shortreg_evals = [(arch_id, shortreg_eval) for arch_id, shortreg_eval in zip(all_arch_ids, all_short_reg_evals)]

    arch_id_reg_evals.sort(key=lambda x: x[1], reverse=True)
    arch_id_shortreg_evals.sort(key=lambda x: x[1], reverse=True)

    assert len(arch_id_reg_evals) == len(arch_id_shortreg_evals)
    
    top_percents = []
    shortreg_ratio_common = []
    for top_percent in top_percent_range:
        top_percents.append(top_percent)
        num_to_keep = int(ma.floor(len(arch_id_reg_evals) * top_percent * 0.01))
        top_percent_arch_id_reg_evals = arch_id_reg_evals[:num_to_keep]
        top_percent_arch_id_shortreg_evals = arch_id_shortreg_evals[:num_to_keep]
        
        # take the set of arch_ids in each method and find overlap with top archs
        set_reg = set([x[0] for x in top_percent_arch_id_reg_evals])
        set_ft = set([x[0] for x in top_percent_arch_id_shortreg_evals])
        ft_num_common = len(set_reg.intersection(set_ft))
        shortreg_ratio_common.append(ft_num_common/num_to_keep)

    # save raw data for other aggregate plots over experiments
    raw_data_dict = {}
    raw_data_dict['top_percents'] = top_percents
    raw_data_dict['spe_shortreg'] = spe_shortreg_top_percents
    raw_data_dict['shortreg_times_avg'] = top_percent_shortreg_times_avg
    raw_data_dict['shortreg_times_std'] = top_percent_shortreg_times_std
    raw_data_dict['shortreg_ratio_common'] = shortreg_ratio_common

    savename = os.path.join(out_dir, 'raw_data.yaml')
    with open(savename, 'w') as f:
        yaml.dump(raw_data_dict, f)

if __name__ == '__main__':
    main()