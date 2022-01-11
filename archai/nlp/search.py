# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Performs evolutionary search to find optimal architectures.
"""

import os
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from archai.common import utils
from archai.nlp.nas.evolution import Evolution, test_evo_search
from archai.nlp.nas.info_getter import get_results
from archai.nlp.nas.nas_utils.jobs_dispatcher import submit_gt_jobs, submit_pareto_front_jobs
from archai.nlp.nas.nas_utils.pareto import compare_w_baseline, get_final_pareto_front, get_gt_pareto


if __name__ == '__main__':
    seed = 1111

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    latency_constraint = {'XeonE5-2690': 0.5,
                          'corei7': 0.5,
                          'corei5': 0.6,
                          'D3_V2': 0.8}

    '''
    #---- orig config
    args = {'default_path': './evo_search','population_size':50, 'parent_size':10, 'mutation_size':20, 'mutation_prob':0.3, 'crossover_size':20, 
            'n_iter':30, 'n_layer_choice':[3,4,5,6,7,8], 'd_model_choice':[128, 256, 512], 'd_inner_choice':list(range(512, 2049, 50))+[2048], 'n_head_choice':[2,4,8],
            'param_constraint':5e6, 'latency_scale':2., 'n_threads':1, 'latency_repeat':5, 'pareto_search':True,
            #--------------- extracting pareto
            'eps':0.05, 'use_convex_hull':False,
            #--------------- brute_force
            'nsamples':20000, 'batch':1000, 'do_train':False,
            #--------------- evaluation scheme  (set start_train to bigger than n_iter to disable training for evaluation)
            'start_train':40, 'train_local':True, 'n_gpus':4, 'gpu_config':'dgx1_4gpu_fp32', 'config_file':'wt103_base.yaml', 'max_step':500, 'experiment_name':'evolution', 
            'scheduler':'constant', 'use_valid':True}
    '''

    # ---- Bigger search-space config
    args = {'default_path': './evo_search',
            'population_size': 100,
            'parent_size': 20,
            'mutation_size': 40,
            'mutation_prob': 0.3,
            'crossover_size': 40,
            'n_iter': 10,
            'n_layer_choice': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'd_model_choice': [128, 256, 512, 650, 800],
            'd_inner_choice': list(range(512, 2049, 50))+list(range(2048, 3072, 200)),
            'n_head_choice': [2, 4, 8],
            'param_constraint': 5e6,
            'latency_scale': 2.,
            'n_threads': 1,
            'latency_repeat': 5,
            'pareto_search': True,
            'device_name': 'XeonE5-2690',
            'eps': 0.05,
            'use_convex_hull': False,
            'nsamples': 20000,
            'batch': 1000,
            'do_train': False,
            'start_train': 40,
            'train_local': True,
            'n_gpus': 1,
            'gpu_config': 'dgx1_1gpu_fp32',
            'config_file': 'wt103_base.yaml',
            'max_step': 500,
            'experiment_name': 'evolution',
            'scheduler': 'constant',
            'use_valid': True}

    args['latency_constraint'] = latency_constraint[args['device_name']]
    path_to_amlt_results = './amlt_logs'

    dir_name = 'param_threshold_{}'.format(args['param_constraint']/1e6)
    
    if args['pareto_search']:
        dir_name += '_pareto'
    if args['use_convex_hull']:
        dir_name += '_convex_hull'
    if args['start_train'] < args['n_iter']:
        dir_name += '_wTrain'

    args['results_path'] = os.path.join(args['default_path'], dir_name+'_'+args['device_name'])

    # choose from {run_search, submit_gt_jobs, extract_pareto, select_pareto, comp_w_baseline, gt_pareto}
    args['phase'] = 'run_search'

    # if True, will use convex hull to extract the final paretos, otherwise, the vanilla pareto formula
    use_convex_hull = False

    # if hybrid is true, takes the pareto on mid-search training, otherwise only looks at nparams
    hybrid = False

    if args['phase'] == 'run_search':
        results_dir = utils.full_path(args['results_path'], create=True)
        with open(os.path.join(results_dir, 'search_config.yaml'), 'w') as f:
            yaml.dump(args, f)

        test_evo_search(args, brute_force=False)

    elif args['phase'] == 'submit_gt_jobs':
        # --------------- submit ground-truth training jobs over the entire population after search
        alg = Evolution(**args)
        submit_gt_jobs(args, alg, max_step=40000, start_config=0, bundle_count=20,
                       n_gpus=8, gpu_config='dgx1_8gpu_fp32', targets=['NLX-NDv2'])

    elif args['phase'] == 'extract_pareto':
        # --------------- extract proxy pareto from all samples seen during the evolutionary search
        ppl_eps = 1  # abosulte ppl difference for extracting the pareto
        param_eps = 0.01  # nomarlized parameter diff for extracting the pareto
        eps = ppl_eps if (args['start_train'] < args['n_iter'] and hybrid) else param_eps
        alg = Evolution(**args)
        get_final_pareto_front(args, alg, eps=eps, hybrid=hybrid, use_convex_hull=use_convex_hull)

    elif args['phase'] == 'select_pareto':
        # --------------- match proxy pareto front points with the baseline and submit selected points on the pareto front for full training
        path_to_baseline = os.path.join(path_to_amlt_results, 'evolution_baselines')

        with open(os.path.join(path_to_baseline, 'params.pkl'), 'rb') as f:
            params_baseline = pickle.load(f)

        with open(os.path.join(path_to_baseline, 'latency_summary_{}.yaml'.format(args['device_name'])), 'r') as f:
            latency_baseline = yaml.safe_load(f)

        baseline_params_list = []
        baseline_latency_list = []

        for k, p in params_baseline.items():
            baseline_params_list.append(p)
            baseline_latency_list.append(latency_baseline[k])

        fname = 'pareto{}'.format('' if hybrid else '_params')
        fname += '_convexHull' if use_convex_hull else ''

        path_to_logs = os.path.join(args['results_path'], fname+'_points.pkl')
        with open(path_to_logs, 'rb') as f:
            pareto = pickle.load(f)

        indices = set()
        keys_to_keep = set()

        for l_b, p_b in zip(baseline_latency_list, baseline_params_list):
            candidate_param = 0
            candidate_latency = np.Inf

            index_param = None
            index_latency = None

            for i, (l, p) in enumerate(zip(pareto['latencies'], pareto['params'])):
                if abs(p-p_b) < 0.01*p_b and l < candidate_latency and l < l_b:
                    index_latency = i
                    candidate_latency = l

                if abs(l-l_b) < 0.01*l_b and p > candidate_param and p > p_b:
                    index_param = i
                    candidate_param = p

            if index_param is not None:
                indices.add(index_param)
                keys_to_keep.add(pareto['keys'][index_param])

            if index_latency is not None:
                indices.add(index_latency)
                keys_to_keep.add(pareto['keys'][index_latency])

        indices = list(indices)

        x_pareto = np.asarray(pareto['params'])
        y_pareto = np.asarray(pareto['latencies']) * 1000.

        plt.figure()
        plt.scatter(x_pareto[indices], y_pareto[indices], s=5, label=k)
        plt.scatter(baseline_params_list, np.asarray(baseline_latency_list)*1000., s=5, label='baseline')
        plt.ylabel('Latency (ms)')
        plt.xlabel('Decoder nParams')
        plt.title('Pareto Curve')
        plt.grid(axis='y')
        plt.legend()
        plt.savefig(os.path.join(args['results_path'], 'search_paretos{}.png'.format('' if hybrid else '_params')), bbox_inches="tight")

        path_to_logs = os.path.join(args['results_path'], fname+'.pkl')
        with open(path_to_logs, 'rb') as f:
            is_pareto_dict = pickle.load(f)

        ks = list(is_pareto_dict.keys())
        for k in ks:
            if not k in keys_to_keep:
                is_pareto_dict[k] = False
        alg = Evolution(**args)
        
        submit_pareto_front_jobs(args, alg, is_pareto_dict, max_step=40000, start_config=0,
                                 bundle_count=20, n_gpus=8, gpu_config='dgx1_8gpu_fp32', targets=['NLX-NDv2'])

    elif args['phase'] == 'comp_w_baseline':
        # --------------- compare baseline with pareto NAS results
        training_exp_name = 'pareto_{}'.format(args['device_name'])
        alg = Evolution(**args)
        compare_w_baseline(args, alg, exp_name=training_exp_name, path_to_dir=path_to_amlt_results,
                           start_config=0, baseline_exp='evolution_baselines', check_pareto=False)

    elif args['phase'] == 'gt_pareto':
        # --------------- compare ground-truth pareto with the proxy pareto
        gt_exp_name = 'evolution_40000'
        os.makedirs(path_to_amlt_results, exist_ok=True)

        command = 'amlt results {} -I "*.json"  -o {} --no-md5'.format(gt_exp_name, path_to_amlt_results)
        os.system(command)
        
        command = 'amlt results {} -I "*.yaml"  -o {} --no-md5'.format(gt_exp_name, path_to_amlt_results)
        os.system(command)
        
        alg = Evolution(**args)
        get_gt_pareto(args, alg, exp_name=gt_exp_name, path_to_dir=path_to_amlt_results, start_config=0,
                      ppl_eps=0.1, latency_eps=0.01, hybrid=hybrid, use_convex_hull=use_convex_hull, min_acceptable_latency_diff=2, baseline_exp=None)  # 'evolution_baselines')
        alg = Evolution(**args)
        compare_w_baseline(args, alg, exp_name=gt_exp_name, path_to_dir=path_to_amlt_results,
                           start_config=0, baseline_exp='evolution_baselines', check_pareto=True)

        # ---------------- compare final val ppl with nparams pareto
        with open('amlt_logs/evolution_40000/params_summary.yaml') as f:
            n_all_params = yaml.load(f)

        gt_results = get_results('evolution_40000', 'amlt_logs/evolution_40000', filetypes=['.json'], verbose=False)

        params_list = []
        val_ppl_list = []

        for job_name, result in gt_results.items():
            if job_name in n_all_params.keys():
                params_list.append(n_all_params[job_name]['FFN'] + n_all_params[job_name]['Attn'])
                val_ppl_list.append(result['valid_perplexity'])

        max_ppl_diff = 0.
        for idx, p in enumerate(params_list):
            for idx2, p2 in enumerate(params_list):
                if abs(p-p2)*100./p < 0.1:
                    if abs(val_ppl_list[idx] - val_ppl_list[idx2]) > max_ppl_diff:
                        max_ppl_diff = abs(val_ppl_list[idx] - val_ppl_list[idx2])
                        max_idx = idx
        
        print(f'maximum vertical difference in val ppl={max_ppl_diff}, happend in p={params_list[idx]}')

        plt.figure()
        plt.scatter(params_list, val_ppl_list, s=5)
        plt.xlabel('# Decoder Params')
        plt.ylabel('Val PPL')
        plt.title('Pareto Curve')
        plt.grid(axis='y')
        fname = 'pareto_params.png'
        plt.savefig(os.path.join(args['results_path'], fname), bbox_inches="tight")
