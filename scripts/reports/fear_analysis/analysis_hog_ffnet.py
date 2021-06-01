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
import pprint

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
from archai.common.analysis_utils import parse_a_nonexprunner_job

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
    #     a = parse_a_nonexprunner_job(job_dir)

    # parallel parsing of yaml logs
    num_workers = 12
    with Pool(num_workers) as p:
        a = p.map(parse_a_nonexprunner_job, job_dirs)

    for storage in a:
        for key, val in storage.items():
            logs[key] = val[0]
            confs[key] = val[1]

    # sometimes a job has not even written logs to yaml
    for key in list(logs.keys()):
        if not logs[key]:
            print(f'arch id {key} did not finish. removing from calculations.')
            logs.pop(key)

    # remove all arch_ids which did not finish
    for key in list(logs.keys()):
        to_delete = False

        # it might have died early
        if 'eval_train' not in list(logs[key].keys()):
            to_delete = True
        
        if to_delete:
            print(f'arch id {key} did not finish. removing from calculations.')
            logs.pop(key)
            continue

        if 'best_train' not in list(logs[key]['eval_train'].keys()):
            print(f'arch id {key} did not finish. removing from calculations.')
            logs.pop(key)
            continue


    lr_bs_best_trains = []
    for key in logs.keys():
        try:
            lr = confs[key]['nas']['eval']['trainer']['optimizer']['lr']
            bs = confs[key]['nas']['eval']['loader']['train_batch']
            best_train = logs[key]['eval_train']['best_train']['top1']
            best_test = logs[key]['eval_train']['best_test']['top1']
            lr_bs_best_trains.append((lr, bs, best_train, best_test))
        except KeyError as err:
            print(f'KeyError {err} not in {key}')

    pprint.pprint(lr_bs_best_trains)
        





if __name__ == '__main__':
    main()