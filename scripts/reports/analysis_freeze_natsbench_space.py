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


def find_valid_log(subdir:str)->str:
    # originally log should be in base folder of eval or search
    logs_filepath_og = os.path.join(str(subdir), 'log.yaml')
    if os.path.isfile(logs_filepath_og):
        return logs_filepath_og
    else:
        # look in the 'dist' folder for any yaml file
        dist_folder = os.path.join(str(subdir), 'dist')
        for f in os.listdir(dist_folder):
            if f.endswith(".yaml"):
                return os.path.join(dist_folder, f)


def parse_a_job(job_dir:str)->Dict:
     if job_dir.is_dir():

        storage = {}
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

            logs_filepath = find_valid_log(subdir)
            config_used_filepath = os.path.join(subdir, 'config_used.yaml')

            if os.path.isfile(logs_filepath):
                fix_yaml(logs_filepath)
                key = job_dir.name + subdir.name + ':' + sub_job
                # parse log
                with open(logs_filepath, 'r') as f:
                    data = yaml.load(f, Loader=yaml.Loader)                    
                # parse config used
                with open(config_used_filepath, 'r') as f:
                    confs = yaml.load(f, Loader=yaml.Loader)
                storage[key] = (data, confs)
            
        return storage


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

    # test single job parsing for debugging
    a = parse_a_job(job_dirs[0])

    # parallel parsing of yaml logs
    with Pool(18) as p:
        a = p.map(parse_a_job, job_dirs)

    for storage in a:
        for key, val in storage.items():
            logs[key] = val[0]
            confs[key] = val[1]
                   
    # examples of accessing logs
    # logs['proxynas_blahblah:eval']['naswotrain_evaluate']['eval_arch']['eval_train']['naswithouttraining']
    # logs['proxynas_blahblah:eval']['regular_evaluate']['regtrainingtop1']
    # logs['proxynas_blahblah:eval']['freeze_evaluate']['eval_arch']['freeze_training']['eval_train']['epochs']['9']['val']['top1']
    # last_epoch_key = list(logs['proxynas_blahblah:eval']['freeze_evaluate']['eval_arch']['freeze_training']['eval_train']['epochs'].keys())[-1]
    # last_val_top1 = logs['proxynas_blahblah:eval']['freeze_evaluate']['eval_arch']['freeze_training']['eval_train']['epochs'][last_epoch_key]['val']['top1']
    # epoch_duration = logs[key]['freeze_evaluate']['eval_arch']['freeze_training']['eval_train']['epochs']['0']['train']['duration']

    # remove all search jobs
    for key in list(logs.keys()):
        if 'search' in key:
            logs.pop(key)

    # remove all arch_ids which did not finish
    for key in list(logs.keys()):
        to_delete = False

        # it might have died early
        if 'freeze_evaluate' not in list(logs[key].keys()):
            to_delete = True

        if 'naswotrain_evaluate' not in list(logs[key].keys()):
            to_delete = True
        
        if 'regular_evaluate' not in list(logs[key].keys()):
            to_delete = True
        
        if to_delete:
            print(f'arch id {key} did not finish. removing from calculations.')
            logs.pop(key)
            continue

        if 'freeze_training'not in list(logs[key]['freeze_evaluate']['eval_arch'].keys()):
            print(f'arch id {key} did not finish. removing from calculations.')
            logs.pop(key)
            continue

        # freeze train may not have finished
        num_freeze_epochs = confs[key]['nas']['eval']['freeze_trainer']['epochs']
        last_freeze_epoch_key = int(list(logs[key]['freeze_evaluate']['eval_arch']['freeze_training']['eval_train']['epochs'].keys())[-1])
        if last_freeze_epoch_key != num_freeze_epochs - 1:
            print(f'arch id {key} did not finish. removing from calculations.')
            logs.pop(key)


    all_reg_evals = []
    
    all_naswotrain_evals = []
    all_freeze_evals_last = []
    all_cond_evals_last = []
    
    all_freeze_flops_last = []
    all_cond_flops_last = []

    all_freeze_time_last = []
    all_cond_time_last = []
    all_partial_time_last = []

    all_freeze_evals = defaultdict(list)

    num_archs_unmet_cond = 0

    for key in logs.keys():
        if 'eval' in key:
            try:

                # if at the end of conditional training train accuracy has not gone above target then don't consider it
                last_cond_epoch_key = list(logs[key]['freeze_evaluate']['eval_arch']['conditional_training']['eval_train']['epochs'].keys())[-1]
                train_end_cond = logs[key]['freeze_evaluate']['eval_arch']['conditional_training']['eval_train']['epochs'][last_cond_epoch_key]['train']['top1']
                if train_end_cond < confs[key]['nas']['eval']['trainer']['train_top1_acc_threshold']:
                    num_archs_unmet_cond += 1
                    continue                

                # freeze evaluation 
                #--------------------

                # at last epoch
                last_freeze_epoch_key = list(logs[key]['freeze_evaluate']['eval_arch']['freeze_training']['eval_train']['epochs'].keys())[-1]
                freeze_eval_top1 = logs[key]['freeze_evaluate']['eval_arch']['freeze_training']['eval_train']['epochs'][last_freeze_epoch_key]['train']['top1']                                            
                all_freeze_evals_last.append(freeze_eval_top1)

                # collect evals at other epochs
                for epoch in range(int(last_freeze_epoch_key)):            
                    all_freeze_evals[epoch].append(logs[key]['freeze_evaluate']['eval_arch']['freeze_training']['eval_train']['epochs'][str(epoch)]['train']['top1'])
        
                # collect flops used for conditional training and freeze training
                freeze_mega_flops_epoch = logs[key]['freeze_evaluate']['eval_arch']['freeze_training']['eval_train']['total_mega_flops_epoch']
                freeze_mega_flops_used = freeze_mega_flops_epoch * int(last_freeze_epoch_key)
                all_freeze_flops_last.append(freeze_mega_flops_used)

                last_cond_epoch_key = list(logs[key]['freeze_evaluate']['eval_arch']['conditional_training']['eval_train']['epochs'].keys())[-1]
                cond_mega_flops_epoch = logs[key]['freeze_evaluate']['eval_arch']['conditional_training']['eval_train']['total_mega_flops_epoch']
                cond_mega_flops_used = cond_mega_flops_epoch * int(last_cond_epoch_key)
                all_cond_flops_last.append(cond_mega_flops_used)

                # collect training error at end of conditional training
                cond_eval_top1 = logs[key]['freeze_evaluate']['eval_arch']['conditional_training']['eval_train']['epochs'][last_cond_epoch_key]['train']['top1']
                all_cond_evals_last.append(cond_eval_top1)

                # collect duration for conditional training and freeze training
                freeze_duration = 0.0
                for epoch_key in logs[key]['freeze_evaluate']['eval_arch']['freeze_training']['eval_train']['epochs']:
                    freeze_duration += logs[key]['freeze_evaluate']['eval_arch']['freeze_training']['eval_train']['epochs'][epoch_key]['train']['duration']

                cond_duration = 0.0
                for epoch_key in logs[key]['freeze_evaluate']['eval_arch']['conditional_training']['eval_train']['epochs']:
                    cond_duration += logs[key]['freeze_evaluate']['eval_arch']['conditional_training']['eval_train']['epochs'][epoch_key]['train']['duration']

                all_freeze_time_last.append(freeze_duration + cond_duration)
                all_cond_time_last.append(cond_duration)
                all_partial_time_last.append(freeze_duration)

                

                # naswotrain
                # --------------
                naswotrain_top1 = logs[key]['naswotrain_evaluate']['eval_arch']['eval_train']['naswithouttraining']
                all_naswotrain_evals.append(naswotrain_top1)

                # regular evaluation
                # --------------------
                reg_eval_top1 = logs[key]['regular_evaluate']['regtrainingtop1']
                all_reg_evals.append(reg_eval_top1)
                    
            except KeyError as err:
                print(f'KeyError {err} not in {key}!')


    # Store some key numbers in results.txt
    results_savename = os.path.join(out_dir, 'results.txt')
    with open(results_savename, 'w') as f:
        f.write(f'Number of archs which did not reach condition: {num_archs_unmet_cond} \n')
        f.write(f'Total valid archs processed: f{len(logs) - num_archs_unmet_cond} \n')

    print(f'Number of archs which did not reach condition: {num_archs_unmet_cond}')
    print(f'Total valid archs processed: {len(logs) - num_archs_unmet_cond}')

    # Sanity check
    assert len(all_reg_evals) == len(all_freeze_evals_last)
    assert len(all_reg_evals) == len(all_cond_evals_last)
    assert len(all_reg_evals) == len(all_naswotrain_evals)
    assert len(all_reg_evals) == len(all_freeze_flops_last)
    assert len(all_reg_evals) == len(all_cond_flops_last)
    assert len(all_reg_evals) == len(all_freeze_time_last)
    
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

    # Conditional training results at last epoch
    cond_tau, cond_p_value = kendalltau(all_reg_evals, all_cond_evals_last)
    cond_spe, cond_sp_value = spearmanr(all_reg_evals, all_cond_evals_last)
    print(f'Conditional Kendall Tau score: {cond_tau:3.03f}, p_value {cond_p_value:3.03f}')
    print(f'Conditional Spearman corr: {cond_spe:3.03f}, p_value {cond_sp_value:3.03f}')
    with open(results_savename, 'a') as f:
        f.write(f'Conditional Kendall Tau score: {cond_tau:3.03f}, p_value {cond_p_value:3.03f} \n')
        f.write(f'Conditional Spearman corr: {cond_spe:3.03f}, p_value {cond_sp_value:3.03f} \n')

    plt.clf()
    sns.scatterplot(x=all_reg_evals, y=all_cond_evals_last)
    plt.xlabel('Test top1 at natsbench full training')
    plt.ylabel('Conditional training')
    plt.grid()
    savename = os.path.join(out_dir, 'proxynas_cond_training_epochs.png')
    plt.savefig(savename, dpi=plt.gcf().dpi, bbox_inches='tight')
 
    # Report average runtime and average flops consumed 
    total_freeze_flops = np.array(all_freeze_flops_last) + np.array(all_cond_flops_last)
    avg_freeze_flops = np.mean(total_freeze_flops)
    std_freeze_flops = np.std(total_freeze_flops)
    stderr_freeze_flops = std_freeze_flops / np.sqrt(len(all_freeze_flops_last))

    avg_freeze_runtime = np.mean(np.array(all_freeze_time_last))
    std_freeze_runtime = np.std(np.array(all_freeze_time_last))
    stderr_freeze_runtime = std_freeze_runtime / np.sqrt(len(all_freeze_time_last))

    avg_cond_runtime = np.mean(np.array(all_cond_time_last))
    std_cond_runtime = np.std(np.array(all_cond_time_last))
    stderr_cond_runtime = std_cond_runtime / np.sqrt(len(all_cond_time_last))

    avg_partial_runtime = np.mean(np.array(all_partial_time_last))
    std_partial_runtime = np.std(np.array(all_partial_time_last))
    stderr_partial_runtime = std_partial_runtime / np.sqrt(len(all_partial_time_last))

    with open(results_savename, 'a') as f:
        f.write(f'Avg. Freeze MFlops: {avg_freeze_flops:.03f}, std {std_freeze_flops}, stderr {stderr_freeze_flops:.03f} \n')
        f.write(f'Avg. Freeze Runtime: {avg_freeze_runtime:.03f}, std {std_freeze_runtime}, stderr {stderr_freeze_runtime:.03f} \n')
        f.write(f'Avg. Conditional Runtime: {avg_cond_runtime:.03f}, std {std_cond_runtime}, stderr {stderr_cond_runtime:.03f} \n')
        f.write(f'Avg. Partial Runtime: {avg_partial_runtime:.03f}, std {std_partial_runtime}, stderr {stderr_partial_runtime:.03f} \n')
 
    # Plot freeze training rank correlations if cutoff at various epochs
    freeze_taus = {}
    freeze_spes = {}
    for epoch_key in all_freeze_evals.keys():
        tau, _ = kendalltau(all_reg_evals, all_freeze_evals[epoch_key])
        spe, _ = spearmanr(all_reg_evals, all_freeze_evals[epoch_key])
        freeze_taus[epoch_key] = tau
        freeze_spes[epoch_key] = spe

    plt.clf()
    for epoch_key in freeze_taus.keys():
        plt.scatter(epoch_key, freeze_taus[epoch_key])
    plt.xlabel('Epochs of freeze training')
    plt.ylabel('Kendall Tau')
    plt.ylim((-1.0, 1.0))
    plt.grid()
    savename = os.path.join(out_dir, 'proxynas_freeze_training_kendall_taus.png')
    plt.savefig(savename, dpi=plt.gcf().dpi, bbox_inches='tight')

    plt.clf()
    for epoch_key in freeze_taus.keys():
        plt.scatter(epoch_key, freeze_spes[epoch_key])
    plt.xlabel('Epochs of freeze training')
    plt.ylabel('Spearman Correlation')
    plt.ylim((-1.0, 1.0))
    plt.grid()
    savename = os.path.join(out_dir, 'proxynas_freeze_training_spearman_corrs.png')
    plt.savefig(savename, dpi=plt.gcf().dpi, bbox_inches='tight')
    

    # Naswottraining results
    naswot_tau, naswot_p_value = kendalltau(all_reg_evals, all_naswotrain_evals)
    naswot_spe, naswot_sp_value = spearmanr(all_reg_evals, all_naswotrain_evals)
    print(f'Naswotraining Kendall Tau score: {naswot_tau:3.03f}, p_value {naswot_p_value:3.03f}')
    print(f'Naswotraining Spearman corr: {naswot_spe:3.03f}, p_value {naswot_sp_value:3.03f}')
    results_savename = os.path.join(out_dir, 'results.txt')
    with open(results_savename, 'a') as f:
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

    plt.clf()
    for epoch_key in freeze_taus.keys():
        plt.scatter(epoch_key, freeze_spes[epoch_key])
    plt.xlabel('Epochs of freeze training')
    plt.ylabel('Spearman Correlation')
    plt.grid()
    savename = os.path.join(out_dir, 'proxynas_freeze_training_spearman_corrs.png')
    plt.savefig(savename, dpi=plt.gcf().dpi, bbox_inches='tight')
    

    # Naswottraining results
    naswot_tau, naswot_p_value = kendalltau(all_reg_evals, all_naswotrain_evals)
    naswot_spe, naswot_sp_value = spearmanr(all_reg_evals, all_naswotrain_evals)
    print(f'Naswotraining Kendall Tau score: {naswot_tau:3.03f}, p_value {naswot_p_value:3.03f}')
    print(f'Naswotraining Spearman corr: {naswot_spe:3.03f}, p_value {naswot_sp_value:3.03f}')
    results_savename = os.path.join(out_dir, 'results.txt')
    with open(results_savename, 'a') as f:
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


    # Rank correlations at top n percent of architectures
    reg_freezelast_naswot_evals = [(all_reg_evals[i], all_freeze_evals_last[i], all_naswotrain_evals[i]) for i in range(len(all_reg_evals))]

    # Rank correlations at top n percent of architectures
    reg_freezelast_naswot_evals = [(all_reg_evals[i], all_freeze_evals_last[i], all_naswotrain_evals[i], all_freeze_time_last[i]) for i in range(len(all_reg_evals))]

    # sort in descending order of accuracy of regular evaluation
    reg_freezelast_naswot_evals.sort(key=lambda x: x[0], reverse=True)

    top_percent_freeze_times_avg = []
    top_percent_freeze_times_std = []

    spe_freeze_top_percents = []
    spe_naswot_top_percents = []
    top_percents = []
    for top_percent in range(10, 101, 10):
        top_percents.append(top_percent)
        num_to_keep = int(ma.floor(len(reg_freezelast_naswot_evals) * top_percent * 0.01))
        top_percent_evals = reg_freezelast_naswot_evals[:num_to_keep]
        top_percent_reg = [x[0] for x in top_percent_evals] 
        top_percent_freeze = [x[1] for x in top_percent_evals]
        top_percent_naswot = [x[2] for x in top_percent_evals]
        top_percent_freeze_times = [x[3] for x in top_percent_evals]

        top_percent_freeze_times_avg.append(np.mean(np.array(top_percent_freeze_times)))
        top_percent_freeze_times_std.append(np.std(np.array(top_percent_freeze_times)))    

        spe_freeze, _ = spearmanr(top_percent_reg, top_percent_freeze)
        spe_freeze_top_percents.append(spe_freeze)
        spe_naswot, _ = spearmanr(top_percent_reg, top_percent_naswot)
        spe_naswot_top_percents.append(spe_naswot)

    plt.clf()
    sns.scatterplot(top_percents, spe_freeze_top_percents)
    sns.scatterplot(top_percents, spe_naswot_top_percents)
    plt.legend(labels=['Freeze Train', 'Naswot'])
    plt.ylim((-1.0, 1.0))
    plt.xlabel('Top percent of architectures')
    plt.ylabel('Spearman Correlation')
    plt.grid()
    savename = os.path.join(out_dir, f'spe_top_archs.png')
    plt.savefig(savename, dpi=plt.gcf().dpi, bbox_inches='tight')

    plt.clf()
    plt.errorbar(top_percents, top_percent_freeze_times_avg, yerr=np.array(top_percent_freeze_times_std)/2, marker='s', mfc='red', ms=10, mew=4)
    plt.xlabel('Top percent of architectures')
    plt.ylabel('Avg. time (s)')
    plt.yticks(np.arange(0,600, step=50))
    plt.grid()
    savename = os.path.join(out_dir, f'freeze_train_duration_top_archs.png')
    plt.savefig(savename, dpi=plt.gcf().dpi, bbox_inches='tight')


if __name__ == '__main__':
    main()