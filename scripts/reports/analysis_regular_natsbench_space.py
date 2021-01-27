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
from analysis_utils import epoch_nodes, fix_yaml, remove_seed_part, group_multi_runs, collect_epoch_nodes, EpochStats, FoldStats, stat2str, get_epoch_stats, get_summary_text, get_details_text, plot_epochs, write_report

import re

def parse_a_job(job_dir:str)->OrderedDict:
     if job_dir.is_dir():
        for subdir in job_dir.iterdir():
            if not subdir.is_dir():
                continue
            # currently we expect that each job was ExperimentRunner job which should have
            # _search or _eval folders
            if subdir.stem.endswith('_search'):
                sub_job = 'search'
            elif subdir.stem.endswith('_eval'):
                sub_job = 'eval'
            else:
                raise RuntimeError(f'Sub directory "{subdir}" in job "{job_dir}" must '
                                'end with either _search or _eval which '
                                'should be the case if ExperimentRunner was used.')

            logs_filepath = os.path.join(str(subdir), 'log.yaml')
            if os.path.isfile(logs_filepath):
                fix_yaml(logs_filepath)
                with open(logs_filepath, 'r') as f:
                    key = job_dir.name + ':' + sub_job
                    data = yaml.load(f, Loader=yaml.Loader)
                return (key, data)


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
    job_dirs = list(results_dir.iterdir())

    # parallel parssing of yaml logs
    with Pool(18) as p:
        a = p.map(parse_a_job, job_dirs)

    for key, data in a:
        logs[key] = data

                   
    # examples of accessing logs
    # best_test = logs[key]['eval_arch']['eval_train']['best_test']['top1']
    # best_train = logs[key]['eval_arch']['eval_train']['best_train']['top1']


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
    reg_shortreg_evals = [(all_reg_evals[i], all_short_reg_evals[i]) for i in range(len(all_reg_evals))]

    # sort in descending order of accuracy of regular evaluation
    reg_shortreg_evals.sort(key=lambda x: x[0], reverse=True)

    spe_shortreg_top_percents = []
    top_percents = []
    for top_percent in range(10, 101, 10):
        top_percents.append(top_percent)
        num_to_keep = int(ma.floor(len(reg_shortreg_evals) * top_percent * 0.01))
        top_percent_evals = reg_shortreg_evals[:num_to_keep]
        top_percent_reg = [x[0] for x in top_percent_evals] 
        top_percent_shortreg = [x[1] for x in top_percent_evals]

        spe_freeze, _ = spearmanr(top_percent_reg, top_percent_shortreg)
        spe_shortreg_top_percents.append(spe_freeze)
                
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

    # time taken
    avg_shortreg_runtime = np.mean(np.array(all_short_reg_time))
    stderr_shortreg_runtime = np.std(np.array(all_short_reg_time)) / np.sqrt(len(all_short_reg_time))

    with open(results_savename, 'a') as f:
        f.write(f'Avg. Shortened Training Runtime: {avg_shortreg_runtime:.03f}, stderr {stderr_shortreg_runtime:.03f} \n')

if __name__ == '__main__':
    main()