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

from scipy.stats import kendalltau, spearmanr

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
from analysis_utils import epoch_nodes, parse_a_job, fix_yaml, remove_seed_part, group_multi_runs, collect_epoch_nodes, EpochStats, FoldStats, stat2str, get_epoch_stats, get_summary_text, get_details_text, plot_epochs, write_report

import re

ZEROCOST_MEASURES = ['grad_norm', 'snip', 'grasp', 'fisher', 'jacob_cov', 'plain', 'synflow', 'synflow_bn']


def main():
    parser = argparse.ArgumentParser(description='Report creator')
    parser.add_argument('--results-dir', '-d', type=str,
                        default=r'~/logdir/proxynas_test_0001',
                        help='folder with experiment results from pt')
    parser.add_argument('--out-dir', '-o', type=str, default=r'~/logdir/reports',
                        help='folder to output reports')
    parser.add_argument('--reg-evals-file', '-r', type=str, default=None,
                        help='optional yaml file which contains full evaluation \
                            of architectures on new datasets not part of natsbench')
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

    # if optional regular evaluation lookup file is provided
    if args.reg_evals_file:
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
    num_workers = 12
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


    all_arch_ids = []
    all_reg_evals = []
    # arch_id -> epoch_num -> measure -> score
    all_zerocost_evals = dict()

    for key in logs.keys():
        if 'eval' in key:
            try:
                
                # regular evaluation 
                # important to get this first since if it is not 
                # available for non-benchmark datasets we need to 
                # remove it from consideration
                # --------------------
                if not args.reg_evals_file:
                    reg_eval_top1 = logs[key]['regular_evaluate']['regtrainingtop1']
                else:
                    # lookup from the provided file since this dataset is not part of the 
                    # benchmark and hence we have to provide the info separately
                    if 'natsbench' in list(confs[key]['nas']['eval'].keys()):
                        arch_id_in_bench = confs[key]['nas']['eval']['natsbench']['arch_index']
                    elif 'nasbench101' in list(confs[key]['nas']['eval'].keys()):
                        arch_id_in_bench = confs[key]['nas']['eval']['nasbench101']['arch_index']
                    
                    if arch_id_in_bench not in list(reg_evals_data.keys()):
                        # if the dataset used is not part of the standard benchmark some of the architectures
                        # may not have full evaluation accuracies available. Remove them from consideration.
                        continue
                    reg_eval_top1 = reg_evals_data[arch_id_in_bench]
                all_reg_evals.append(reg_eval_top1)

                # record the arch id
                # --------------------
                if 'natsbench' in list(confs[key]['nas']['eval'].keys()):
                    arch_id = confs[key]['nas']['eval']['natsbench']['arch_index']
                    all_arch_ids.append(arch_id)
                elif 'nasbench101' in list(confs[key]['nas']['eval'].keys()):
                    arch_id = confs[key]['nas']['eval']['nasbench101']['arch_index']
                    all_arch_ids.append(arch_id)

                # zerocost scores
                #-------------------------------
                epoch_num_measures = OrderedDict()
                for tkey in logs[key]['zerocost_evaluate']['eval_arch'].keys():
                    if 'zerocost' in tkey:
                        epoch_num = int(tkey.split('_')[-1])
                        this_epoch_measures = OrderedDict()
                        for measure in ZEROCOST_MEASURES:
                            score = logs[key]['zerocost_evaluate']['eval_arch'][tkey]['eval_train'][measure]
                            # jacob_cov tends to produce nans sometimes
                            if ma.isnan(score):
                                score = 0.0
                            this_epoch_measures[measure] = score
                        epoch_num_measures[epoch_num] = this_epoch_measures
                all_zerocost_evals[arch_id] = epoch_num_measures

            except KeyError as err:
                print(f'KeyError {err} not in {key}')
                sys.exit()    

    # Store some key numbers in results.txt
    results_savename = os.path.join(out_dir, 'results.txt')
    with open(results_savename, 'w') as f:
        f.write(f'Total valid archs processed: {len(all_reg_evals)} \n')

    print(f'Total valid archs processed: {len(all_reg_evals)}')

    # Sanity check
    assert len(all_reg_evals) == len(all_arch_ids)
    assert len(all_reg_evals) == len(all_zerocost_evals)

    # make sure that all arch_ids have same number of epochs
    akey = list(all_zerocost_evals.keys())[0]
    num_epochs = len(all_zerocost_evals[akey])
    for arch_id in all_zerocost_evals.keys():
        assert num_epochs == len(all_zerocost_evals[arch_id])


    # for each epoch num compute Spearman's correlation
    # for each measure
    # measure -> epoch_num -> spe
    measures_res = OrderedDict()
    for measure in ZEROCOST_MEASURES:
        epoch_num_spe = OrderedDict()
        for epoch_num in range(num_epochs):                
            all_scores = []
            for arch_id in all_zerocost_evals:
                score = all_zerocost_evals[arch_id][epoch_num][measure]
                all_scores.append(score)
            
            assert len(all_reg_evals) == len(all_scores)
            spe, _ = spearmanr(all_reg_evals, all_scores)
            epoch_num_spe[epoch_num] = spe        
        measures_res[measure] = epoch_num_spe
    
    # plot
    fig = go.Figure()
    for measure in ZEROCOST_MEASURES:
        xs = [key for key in measures_res[measure].keys()]
        ys = [measures_res[measure][epoch_num] for epoch_num in measures_res[measure].keys()]
        
        fig.add_trace(go.Scatter(x=xs, y=ys, name=measure, mode='markers+lines', showlegend=True))
        
    fig.update_layout(xaxis_title='Epochs',
                    yaxis_title='Spearman Corr.')
    fig.update_layout(font=dict(size=48))

    savename_html = os.path.join(out_dir, 'zerocost_epochs_vs_spe.html')
    fig.write_html(savename_html)

    savename_pdf = os.path.join(out_dir, 'zerocost_epochs_vs_spe.pdf')
    fig.write_image(savename_pdf, engine="kaleido", width=1500, height=750, scale=1)

    fig.show()









if __name__ == '__main__':
    main()