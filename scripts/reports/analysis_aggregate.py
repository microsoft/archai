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
from multiprocessing import Pool

from runstats import Statistics

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from archai.common import utils
from archai.common.ordereddict_logger import OrderedDictLogger

from archai.common.analysis_utils import epoch_nodes, fix_yaml, remove_seed_part, group_multi_runs, collect_epoch_nodes, EpochStats, FoldStats, stat2str, get_epoch_stats, get_summary_text, get_details_text, plot_epochs, write_report

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
    job_count = 0
    job_dirs = list(results_dir.iterdir())

    # parallel parssing of yaml logs
    with Pool(18) as p:
        a = p.map(parse_a_job, job_dirs)

    for key, data in a:
        logs[key] = data
        job_count += 1

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

    
    # create list of epoch nodes having same path in the logs
    grouped_logs = group_multi_runs(logs)
    collated_grouped_logs = collect_epoch_nodes(grouped_logs)
    summary_text, details_text = '', ''

    for log_key, grouped_logs in collated_grouped_logs.items():
        # for each path for epochs nodes, compute stats
        for node_path, logs_epochs_nodes in grouped_logs.items():
            collated_epoch_stats = get_epoch_stats(node_path, logs_epochs_nodes)
            summary_text += get_summary_text(log_key, out_dir, node_path, collated_epoch_stats, len(logs_epochs_nodes))
            details_text += get_details_text(log_key, out_dir, node_path, collated_epoch_stats, len(logs_epochs_nodes))

    write_report('summary.md', **vars())
    write_report('details.md', **vars())


if __name__ == '__main__':
    main()