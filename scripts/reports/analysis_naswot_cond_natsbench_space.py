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

def myfunc(x):
    return x*x

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

    # # single process parsing of yaml logs
    # for job_dir in tqdm(results_dir.iterdir()):
    #     if job_dir.is_dir():
    #         for subdir in job_dir.iterdir():
    #             if not subdir.is_dir():
    #                 continue
    #             # currently we expect that each job was ExperimentRunner job which should have
    #             # _search or _eval folders
    #             if subdir.stem.endswith('_search'):
    #                 sub_job = 'search'
    #             elif subdir.stem.endswith('_eval'):
    #                 sub_job = 'eval'
    #             else:
    #                 raise RuntimeError(f'Sub directory "{subdir}" in job "{job_dir}" must '
    #                                 'end with either _search or _eval which '
    #                                 'should be the case if ExperimentRunner was used.')

    #             logs_filepath = os.path.join(str(subdir), 'log.yaml')
    #             if os.path.isfile(logs_filepath):
    #                 fix_yaml(logs_filepath)
    #                 with open(logs_filepath, 'r') as f:
    #                     key = job_dir.name + ':' + sub_job
    #                     logs[key] = yaml.load(f, Loader=yaml.Loader)
                        

    # examples of accessing logs
    # logs['proxynas_blahblah:eval']['naswotrain_evaluate']['eval_arch']['eval_train']['naswithouttraining']
    # logs['proxynas_blahblah:eval']['regular_evaluate']['regtrainingtop1']
    # logs['proxynas_blahblah:eval']['naswot_cond_evaluate']['eval_arch']['conditional_naswot']['eval_train']['naswithouttraining']
    

    all_reg_evals = []
    all_naswotrain_evals = []
    all_cond_naswotrain_evals = []
    
    for key in logs.keys():
        if 'eval' in key:
            try:
                # naswotrain
                naswotrain_top1 = logs[key]['naswotrain_evaluate']['eval_arch']['eval_train']['naswithouttraining']
                all_naswotrain_evals.append(naswotrain_top1)

                # regular evaluation
                reg_eval_top1 = logs[key]['regular_evaluate']['regtrainingtop1']
                all_reg_evals.append(reg_eval_top1)

                # conditional naswotrain
                #------------------------
                cond_naswot = logs[key]['naswot_cond_evaluate']['eval_arch']['conditional_naswot']['eval_train']['naswithouttraining']
                all_cond_naswotrain_evals.append(cond_naswot)                
                    
            except KeyError as err:
                print(f'KeyError {err} not in {key}!')

        
    # Naswottraining results
    naswot_tau, naswot_p_value = kendalltau(all_reg_evals, all_naswotrain_evals)
    naswot_spe, naswot_sp_value = spearmanr(all_reg_evals, all_naswotrain_evals)
    print(f'Naswotraining Kendall Tau score: {naswot_tau:3.03f}, p_value {naswot_p_value:3.03f}')
    print(f'Naswotraining Spearman corr: {naswot_spe:3.03f}, p_value {naswot_sp_value:3.03f}')
    results_savename = os.path.join(out_dir, 'results.txt')
    with open(results_savename, 'w') as f:
        f.write(f'Naswotraining Kendall Tau score: {naswot_tau:3.03f}, p_value {naswot_p_value:3.03f} \n')
        f.write(f'Naswotraining Spearman corr: {naswot_spe:3.03f}, p_value {naswot_sp_value:3.03f} \n')

    plt.clf()
    sns.scatterplot(all_reg_evals, all_naswotrain_evals)
    plt.xlabel('Test top1 at 200 epochs')
    plt.ylabel('Naswotraining')
    plt.title('Naswotraining')
    plt.grid()
    savename = os.path.join(out_dir, 'proxynas_naswotraining.png')
    plt.savefig(savename, dpi=plt.gcf().dpi, bbox_inches='tight')

    # Conditional naswot results
    cond_naswot_tau, cond_naswot_p_value = kendalltau(all_reg_evals, all_cond_naswotrain_evals)
    cond_naswot_spe, cond_naswot_sp_value = spearmanr(all_reg_evals, all_cond_naswotrain_evals)
    print(f'Cond. Naswot Kendall Tau score: {cond_naswot_tau:3.03f}, p_value {cond_naswot_p_value:3.03f}')
    print(f'Cond. Naswot Spearman corr: {cond_naswot_spe:3.03f}, p_value {cond_naswot_sp_value:3.03f}')
    results_savename = os.path.join(out_dir, 'results.txt')
    with open(results_savename, 'w') as f:
        f.write(f'Cond. Naswot Kendall Tau score: {cond_naswot_tau:3.03f}, p_value {cond_naswot_p_value:3.03f} \n')
        f.write(f'Cond. Naswot Spearman corr: {cond_naswot_spe:3.03f}, p_value {cond_naswot_sp_value:3.03f} \n')

    plt.clf()
    sns.scatterplot(all_reg_evals, all_cond_naswotrain_evals)
    plt.xlabel('Test top1 at 200 epochs')
    plt.ylabel('Cond. Naswot')
    plt.title('Cond. Naswot')
    plt.grid()
    savename = os.path.join(out_dir, 'proxynas_condnaswot.png')
    plt.savefig(savename, dpi=plt.gcf().dpi, bbox_inches='tight')

    # Rank correlations at top n percent of architectures
    assert len(all_reg_evals) == len(all_naswotrain_evals)
    assert len(all_reg_evals) == len(all_cond_naswotrain_evals)
    reg_cnaswot_naswot_evals = [(all_reg_evals[i], all_cond_naswotrain_evals[i], all_naswotrain_evals[i]) for i in range(len(all_reg_evals))]

    # sort in descending order of accuracy of regular evaluation
    reg_cnaswot_naswot_evals.sort(key=lambda x: x[0], reverse=True)

    spe_cnaswot_top_percents = []
    spe_naswot_top_percents = []
    top_percents = []
    for top_percent in range(10, 101, 10):
        top_percents.append(top_percent)
        num_to_keep = int(ma.floor(len(reg_cnaswot_naswot_evals) * top_percent * 0.01))
        top_percent_evals = reg_cnaswot_naswot_evals[:num_to_keep]
        top_percent_reg = [x[0] for x in top_percent_evals] 
        top_percent_cnaswot = [x[1] for x in top_percent_evals]
        top_percent_naswot = [x[2] for x in top_percent_evals]

        spe_cnaswot, _ = spearmanr(top_percent_reg, top_percent_cnaswot)
        spe_cnaswot_top_percents.append(spe_cnaswot)
        spe_naswot, _ = spearmanr(top_percent_reg, top_percent_naswot)
        spe_naswot_top_percents.append(spe_naswot)

    plt.clf()
    sns.scatterplot(top_percents, spe_cnaswot_top_percents)
    sns.scatterplot(top_percents, spe_naswot_top_percents)
    plt.legend(labels=['Conditional Naswot', 'Naswot'])
    plt.ylim((0.0, 1.0))
    plt.xlabel('Top percent of architectures')
    plt.ylabel('Spearman Correlation')
    plt.grid()
    savename = os.path.join(out_dir, f'spe_top_archs.png')
    plt.savefig(savename, dpi=plt.gcf().dpi, bbox_inches='tight')

if __name__ == '__main__':
    main()