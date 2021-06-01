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
    num_workers = 12
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


    archid_testacc = {}
    
    for key in logs.keys():
        if 'eval' in key:
            try:                
                best_test = logs[key]['eval_arch']['eval_train']['best_test']['top1']
                            
                # record the arch id
                # --------------------
                if 'natsbench' in list(confs[key]['nas']['eval'].keys()):
                    archid = confs[key]['nas']['eval']['natsbench']['arch_index']                    
                elif 'nasbench101' in list(confs[key]['nas']['eval'].keys()):
                    archid = confs[key]['nas']['eval']['nasbench101']['arch_index']

                archid_testacc[archid] = best_test
                                
            except KeyError as err:
                print(f'KeyError {err} not in {key}!')

    # dump to yaml
    savename = os.path.join(out_dir, 'archid_testacc.yaml')
    with open(savename, 'w') as f:
        yaml.dump(archid_testacc, f)

    print(f'Wrote benchmark file with {len(archid_testacc)} archs')

    

    
    

if __name__ == '__main__':
    main()