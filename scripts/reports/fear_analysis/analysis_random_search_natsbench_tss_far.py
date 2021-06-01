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

from scipy.stats import kendalltau, spearmanr, sem
import statistics

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


def find_best_test_sofar(best_tests:List[Tuple[int, float]], arch_ids_so_far:List[int])->float:
    arch_ids_in_best_test = [x[0] for x in best_tests]

    # go through ids so far in reverse order
    # until you find a hit in best_tests
    for id in reversed(arch_ids_so_far):
        if id in arch_ids_in_best_test:
            for aid, test in best_tests:
                if aid == id:
                    return test


def main():
    parser = argparse.ArgumentParser(description='Report creator')
    parser.add_argument('--results-dir', '-d', type=str,
                        default=r'~/logdir/random_search_far',
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

    raw_data = {}

    for key in logs.keys():
        # Get total duration of the run
        # which is the sum of all conditional and freeze
        # trainings over all architectures
        duration = 0.0

        best_tests = None
        best_trains = None

        duration_best_after_each_model = []
        arch_ids_touched = []

        archs_touched_counter = 0
        for skey in logs[key].keys():
            if 'conditional' in skey or 'freeze' in skey:
                if 'conditional' in skey:
                    train_key = 'arch_train'
                else:
                    train_key = 'eval_train'
                for ekey in logs[key][skey][train_key]['epochs'].keys():
                    eduration = logs[key][skey][train_key]['epochs'][ekey]['train']['duration']
                    duration += eduration
                arch_id = int(skey.split('_')[-1])
                if arch_id not in arch_ids_touched:
                    archs_touched_counter += 1
                    arch_ids_touched.append(arch_id)
                    # after touching this model record the
                    # time taken so far and the test accuracy of 
                    # the best model found so far by random search
                    if best_tests is not None:
                        best_test_sofar = find_best_test_sofar(best_tests, arch_ids_touched)
                        duration_best_after_each_model.append((duration, best_test_sofar))

            # Get the last best_trains_tests
            # NOTE: that due to references being logged
            # the entire run's data is logged after every
            # architecture. So this is redundant but for safety.
            if 'best_trains_tests' in skey:
                best_trains = logs[key][skey]['best_trains']
                best_tests = logs[key][skey]['best_tests']
                assert len(best_trains) == len(best_tests)

        # find the test error of the best train 
        best_test = best_tests[-1][1]

        raw_data[key] = (duration, best_test, duration_best_after_each_model)

    run_durations = [raw_data[key][0] for key in raw_data.keys()]
    max_accs = [raw_data[key][1] for key in raw_data.keys()]
    trajs = [raw_data[key][2] for key in raw_data.keys()]

    avg_duration = statistics.mean(run_durations)
    stderr_duration = sem(np.array(run_durations))

    avg_max_acc = statistics.mean(max_accs)
    stderr_max_acc = sem(np.array(max_accs))

    data_to_save = {}
    data_to_save['avg_duration'] = avg_duration
    data_to_save['stderr_duration'] = stderr_duration
    data_to_save['avg_max_acc'] = avg_max_acc
    data_to_save['stderr_max_acc'] = stderr_max_acc
    data_to_save['trajs'] = trajs

    savename = os.path.join(out_dir, 'raw_data.yaml')
    with open(savename, 'w') as f:
        yaml.dump(data_to_save, f)

if __name__ == '__main__':
    main()