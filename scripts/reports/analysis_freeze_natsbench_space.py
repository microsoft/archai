# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from typing import Dict, List, Type, Iterator, Tuple
import glob
import os
import pathlib
from collections import OrderedDict
import yaml
from inspect import getsourcefile
import re
from tqdm import tqdm
import seaborn as sns

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
    with Pool(24) as p:
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
    # logs['proxynas_blahblah:eval']['freeze_evaluate']['eval_arch']['freeze_training']['eval_train']['epochs']['9']['val']['top1']
    # last_epoch_key = list(logs['proxynas_blahblah:eval']['freeze_evaluate']['eval_arch']['freeze_training']['eval_train']['epochs'].keys())[-1]
    # last_val_top1 = logs['proxynas_blahblah:eval']['freeze_evaluate']['eval_arch']['freeze_training']['eval_train']['epochs'][last_epoch_key]['val']['top1']

    all_reg_evals = []
    all_naswotrain_evals = []
    all_freeze_evals = []
    for key in logs.keys():
        if 'eval' in key:
            try:
                # naswotrain
                naswotrain_top1 = logs[key]['naswotrain_evaluate']['eval_arch']['eval_train']['naswithouttraining']

                # regular evaluation
                reg_eval_top1 = logs[key]['regular_evaluate']['regtrainingtop1']

                # freeze evaluationj
                last_epoch_key = list(logs[key]['freeze_evaluate']['eval_arch']['freeze_training']['eval_train']['epochs'].keys())[-1]
                freeze_eval_top1 = logs[key]['freeze_evaluate']['eval_arch']['freeze_training']['eval_train']['epochs'][last_epoch_key]['val']['top1']
                
                all_naswotrain_evals.append(naswotrain_top1)
                all_reg_evals.append(reg_eval_top1)
                all_freeze_evals.append(freeze_eval_top1)
            except KeyError as err:
                print(f'KeyError {err} in {key}')

    # Freeze training results        
    freeze_tau, freeze_p_value = kendalltau(all_reg_evals, all_freeze_evals)
    freeze_spe, freeze_sp_value = spearmanr(all_reg_evals, all_freeze_evals)
    print(f'Freeze Kendall Tau score: {freeze_tau}, p_value {freeze_p_value}')
    print(f'Freeze Spearman corr: {freeze_spe}, p_value {freeze_sp_value}')
    results_savename = os.path.join(out_dir, 'results.txt')
    with open(results_savename, 'w') as f:
        f.write(f'Freeze Kendall Tau score: {freeze_tau}, p_value {freeze_p_value} \n')
        f.write(f'Freeze Spearman corr: {freeze_spe}, p_value {freeze_sp_value} \n')


    sns.scatterplot(x=all_reg_evals, y=all_freeze_evals)
    plt.xlabel('Test top1 at natsbench full training')
    plt.ylabel('Freeze training')
    plt.title('Freeze training at 0.60 val')
    savename = os.path.join(out_dir, 'proxynas_0.7_freeze_training_100_epochs.png')
    plt.savefig(savename, dpi=plt.gcf().dpi, bbox_inches='tight')

    # Naswottraining results
    naswot_tau, naswot_p_value = kendalltau(all_reg_evals, all_naswotrain_evals)
    naswot_spe, naswot_sp_value = spearmanr(all_reg_evals, all_naswotrain_evals)
    print(f'Naswotraining Kendall Tau score: {naswot_tau}, p_value {naswot_p_value} \n')
    print(f'Naswotraining Spearman corr: {naswot_spe}, p_value {naswot_sp_value} \n')
    results_savename = os.path.join(out_dir, 'results.txt')
    with open(results_savename, 'a') as f:
        f.write(f'Naswotraining Kendall Tau score: {naswot_tau}, p_value {naswot_p_value}')
        f.write(f'Naswotraining Spearman corr: {naswot_spe}, p_value {naswot_sp_value}')

    
    plt.scatter(all_reg_evals, all_naswotrain_evals)
    plt.xlabel('Val top1 at 600 epochs')
    plt.ylabel('Naswotraining')
    plt.title('Naswotraining')
    savename = os.path.join(out_dir, 'proxynas_naswotraining.png')
    plt.savefig(savename, dpi=plt.gcf().dpi, bbox_inches='tight')




if __name__ == '__main__':
    main()