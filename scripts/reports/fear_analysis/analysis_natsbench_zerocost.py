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

from scipy.stats import kendalltau, spearmanr, pearsonr

from runstats import Statistics

#import matplotlib
#matplotlib.use('TkAgg')
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from collections import namedtuple
from itertools import product


from archai.common import utils
from archai.common.ordereddict_logger import OrderedDictLogger
from archai.common.analysis_utils import epoch_nodes, parse_a_job, fix_yaml, remove_seed_part, group_multi_runs, collect_epoch_nodes, EpochStats, FoldStats, stat2str, get_epoch_stats, get_summary_text, get_details_text, plot_epochs, write_report

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
    parser.add_argument('--params-flops-file', '-p', type=str, default=None,
                        help='optional yaml file which contains params flops information \
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
    reg_evals_data = None
    if args.reg_evals_file:
        with open(args.reg_evals_file, 'r') as f:
            reg_evals_data = yaml.load(f, Loader=yaml.Loader)
    
    # if optional params flops lookup file is provided
    params_flops_data = None
    if args.params_flops_file:
        with open(args.params_flops_file, 'r') as f:
            params_flops_data = yaml.load(f, Loader=yaml.Loader)

    # get list of all structured logs for each job
    logs = {}
    confs = {}
    job_dirs = list(results_dir.iterdir())

    # # test single job parsing for debugging
    # # WARNING: very slow, just use for debugging
    # for job_dir in job_dirs:
    #     a = parse_a_job(job_dir)

    # parallel parsing of yaml logs
    num_workers = 48
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

    # remove all arch_ids which did not finish
    for key in list(logs.keys()):
        to_delete = False

        # it might have died early
        if 'zerocost_evaluate' not in list(logs[key].keys()):
            to_delete = True
    
        if to_delete:
            print(f'arch id {key} did not finish. removing from calculations.')
            logs.pop(key)
            continue


    all_arch_ids = []
    all_reg_evals = []
    all_params_evals = []
    all_flops_evals = []
    all_zerocost_init_evals = defaultdict(list)

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
                    elif 'dartsspace' in list(confs[key]['nas']['eval'].keys()):
                        arch_id_in_bench = confs[key]['nas']['eval']['dartsspace']['arch_index']
                    
                    if arch_id_in_bench not in list(reg_evals_data.keys()):
                        # if the dataset used is not part of the standard benchmark some of the architectures
                        # may not have full evaluation accuracies available. Remove them from consideration.
                        continue
                    reg_eval_top1 = reg_evals_data[arch_id_in_bench]

                    if params_flops_data:
                        params = params_flops_data[arch_id_in_bench]['params']
                        flops = params_flops_data[arch_id_in_bench]['flops']
                        all_params_evals.append(params)
                        all_flops_evals.append(flops)
    
                all_reg_evals.append(reg_eval_top1)
                

                # zerocost initial scores
                #-------------------------------
                for measure in ZEROCOST_MEASURES:
                    score = logs[key]['zerocost_evaluate']['eval_arch']['eval_train'][measure]
                    all_zerocost_init_evals[measure].append(score)

                # record the arch id
                # --------------------
                if 'natsbench' in list(confs[key]['nas']['eval'].keys()):
                    all_arch_ids.append(confs[key]['nas']['eval']['natsbench']['arch_index'])
                elif 'nasbench101' in list(confs[key]['nas']['eval'].keys()):
                    all_arch_ids.append(confs[key]['nas']['eval']['nasbench101']['arch_index'])
                elif 'dartsspace' in list(confs[key]['nas']['eval'].keys()):
                    all_arch_ids.append(confs[key]['nas']['eval']['dartsspace']['arch_index'])
                
            except KeyError as err:
                print(f'KeyError {err} not in {key}')
                sys.exit()

    # Sanity check
    for measure in ZEROCOST_MEASURES:
        assert len(all_reg_evals) == len(all_zerocost_init_evals[measure])
    assert len(all_reg_evals) == len(all_arch_ids)
    
    # if params flops is present compute spearman wrt params and flops
    # also compute scatter plots for params vs synflow
    if params_flops_data:
        assert len(all_reg_evals) == len(all_params_evals)
        assert len(all_reg_evals) == len(all_flops_evals)

        spe_params, _ = spearmanr(all_reg_evals, all_params_evals)
        spe_flops, _ = spearmanr(all_reg_evals, all_flops_evals)

        # Store some key numbers in results.txt
        results_savename = os.path.join(out_dir, 'results.txt')
        with open(results_savename, 'w') as f:
            f.write(f'Total valid archs processed: {len(all_reg_evals)} \n')
            f.write(f'Spearman wrt params: {spe_params} \n')
            f.write(f'Spearman wrt flops: {spe_flops} \n')

        print(f'Total valid archs processed: {len(all_reg_evals)}')
        print(f'Spearman wrt params: {spe_params} \n')
        print(f'Spearman wrt flops: {spe_flops} \n')

        # scatter params vs. synflow
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=all_params_evals, y=all_zerocost_init_evals['synflow'], mode='markers'))
        fig.update_layout(xaxis_title="Parameters",
                        yaxis_title="Synflow")
        fig.update_layout(font=dict(size=36)) # font size
        fig.update_traces(marker=dict(size=20)) # marker size
        savename_html = os.path.join(out_dir, f'params_vs_synflow.html')
        savename_png = os.path.join(out_dir, f'params_vs_synflow.png')
        fig.write_html(savename_html)
        fig.write_image(savename_png, width=1500, height=1500, scale=1)

        # create heatmap of all pairs of proxies along with params, flops, gt
        ZEROCOST_MEASURES_PF = ['grad_norm', 'snip', 'grasp', 'fisher', 'jacob_cov', 'synflow', 'params', 'flops', 'gt']
        all_zerocost_init_evals['params'] = all_params_evals
        all_zerocost_init_evals['flops'] = all_flops_evals
        all_zerocost_init_evals['gt'] = all_reg_evals
        hm = np.zeros((len(ZEROCOST_MEASURES_PF), len(ZEROCOST_MEASURES_PF))) 
        for i, m1 in enumerate(ZEROCOST_MEASURES_PF):
            for j, m2 in enumerate(ZEROCOST_MEASURES_PF):
                # sometimes jacob_cov has a nan here and there. ignore those.
                m1_scores = all_zerocost_init_evals[m1]
                m2_scores = all_zerocost_init_evals[m2]
                valid_scores = [x for x in zip(m1_scores, m2_scores) if not ma.isnan(x[0]) and not ma.isnan(x[1])]
                m1_valid = [x[0] for x in valid_scores]
                m2_valid = [x[1] for x in valid_scores]
                spe, _ = spearmanr(m1_valid, m2_valid)
                hm[i][j] = spe

        fig = px.imshow(hm, text_auto="0.1f", x=ZEROCOST_MEASURES_PF, y=ZEROCOST_MEASURES_PF)
        fig.update_layout(font=dict(size=36)) # font size
        savename_html = os.path.join(out_dir, f'all_pairs_zc_spe.html')
        savename_png = os.path.join(out_dir, f'all_pairs_zc_spe.png')
        fig.write_html(savename_html)
        fig.write_image(savename_png, width=1500, height=1500, scale=1)






    results_savename = os.path.join(out_dir, 'results.txt')
    with open(results_savename, 'a') as f:
        f.write(f'Total valid archs processed: {len(all_reg_evals)} \n')

    top_percent_range = range(2, 101, 2)
    # Rank correlations at top n percent of architectures
    # -------------------------------------------------------
    spe_top_percents_init = defaultdict(list)    

    for measure in ZEROCOST_MEASURES:
        reg_init = [(all_reg_evals[i], all_zerocost_init_evals[measure][i]) for i in range(len(all_reg_evals))]
        reg_init.sort(key=lambda x: x[0], reverse=True)

        top_percents = []
        
        for top_percent in top_percent_range:
            top_percents.append(top_percent)
            num_to_keep = int(ma.floor(len(reg_init) * top_percent * 0.01))
            top_percent_evals = reg_init[:num_to_keep]
            # sometimes jacob_cov has a nan here and there. ignore those.
            top_percent_reg = [x[0] for x in top_percent_evals 
                                if not ma.isnan(x[0]) and not ma.isnan(x[1])]
            top_percent_init = [x[1] for x in top_percent_evals 
                                if not ma.isnan(x[0]) and not ma.isnan(x[1])]

            assert len(top_percent_reg) == len(top_percent_init)

            spe_init, _ = spearmanr(top_percent_reg, top_percent_init)
            # for the entire bin of archs scatter plot 
            # groundtruth accuracy (x-axis) vs. measure and save 
            if top_percent == 100:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=top_percent_reg, y=top_percent_init, mode='markers'))
                fig.update_layout(xaxis_title="Test Accuracy",
                                yaxis_title=f"{measure}")
                fig.update_layout(font=dict(size=36)) # font size
                fig.update_traces(marker=dict(size=20)) # marker size
                savename_html = os.path.join(out_dir, f'test_accuracy_vs_{measure}.html')
                savename_png = os.path.join(out_dir, f'test_accuracy_vs_{measure}.png')
                fig.write_html(savename_html)
                fig.write_image(savename_png, width=1500, height=1500, scale=1)
                #fig.show()


            spe_top_percents_init[measure].append(spe_init)

        spe_top_percents_init['top_percents'] = top_percents
        

    # overlap in top x% of architectures between method and groundtruth
    # ------------------------------------------------------------------
    cr_init_top_percents = defaultdict(list)

    arch_id_reg_evals = [(all_arch_ids[i], all_reg_evals[i]) for i in range(len(all_reg_evals))]
    arch_id_reg_evals.sort(key=lambda x: x[1], reverse=True)
    
    for measure in ZEROCOST_MEASURES:
        arch_id_init = [(all_arch_ids[i], all_zerocost_init_evals[measure][i]) for i in range(len(all_reg_evals))]
        arch_id_init.sort(key=lambda x: x[1], reverse=True)

        assert len(arch_id_reg_evals) == len(arch_id_init)
    
        top_percents = []
        
        for top_percent in top_percent_range:
            top_percents.append(top_percent)
            num_to_keep = int(ma.floor(len(arch_id_reg_evals) * top_percent * 0.01))
            top_percent_arch_id_reg_evals = arch_id_reg_evals[:num_to_keep]
            top_percent_arch_id_init_evals = arch_id_init[:num_to_keep]            

            # take the set of arch_ids in each method and find overlap with top archs
            set_reg = set([x[0] for x in top_percent_arch_id_reg_evals])
            set_init = set([x[0] for x in top_percent_arch_id_init_evals])           
            init_num_common = len(set_reg.intersection(set_init))
            cr_init_top_percents[measure].append(init_num_common/num_to_keep)
            
        cr_init_top_percents['top_percents'] = top_percents
        
    # save data                    
    save_data(spe_top_percents_init, cr_init_top_percents, out_dir)

    # print out summary in pretty format to text file
    with open(results_savename, 'a') as f:
        f.write('spearman correlations: \n')
        for measure in ZEROCOST_MEASURES:
            f.write(f'{measure}: {spe_top_percents_init[measure][-1]} \n')
            print(f'{measure}: {spe_top_percents_init[measure][-1]} \n')
    


def save_data(spe_top_percents:List[float], cr_top_percents:List[float], savefolder:str):
    # save raw data for other aggregate plots over experiments
    # --------------------------------------------------------
    raw_data_dict = {}
    raw_data_dict['top_percents'] = spe_top_percents['top_percents']

    for measure in ZEROCOST_MEASURES:
        raw_data_dict[measure+'_spe'] = spe_top_percents[measure]
        raw_data_dict[measure+'_ratio_common'] = cr_top_percents[measure]

    os.makedirs(savefolder, exist_ok=True)
    savename = os.path.join(savefolder, 'raw_data.yaml')
    with open(savename, 'w') as f:
        yaml.dump(raw_data_dict, f)



if __name__ == '__main__':
    main()