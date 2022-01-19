# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Performs evolutionary search to find optimal architectures.
"""

import argparse
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from archai.common import utils
from archai.nlp.nas.evolution import Evolution, run_search
from archai.nlp.nas.nas_utils.dispatcher import (submit_ground_truth_jobs,
                                                 submit_pareto_front_jobs)
from archai.nlp.nas.nas_utils.pareto_front import (compare_pareto_fronts,
                                                   find_final_pareto_front,
                                                   find_ground_truth_pareto)
from archai.nlp.nas.nas_utils.parser import parse_results_from_experiment


def parse_args():
    parser = argparse.ArgumentParser(description='Evolutionary search with language models.')

    parser.add_argument('--default_path',
                        type=str,
                        default='./evo_search',
                        help='Path to the folder that will save the results.')

    parser.add_argument('--phase',
                        type=str,
                        default='run_search',
                        choices=['run_search', 'submit_gt_jobs',
                                 'extract_pareto', 'select_pareto',
                                 'compare_pareto', 'gt_pareto'],
                        help='Search phase.')

    parser.add_argument('--population_size',
                        type=int,
                        default=100,
                        help='Size of the population.')

    parser.add_argument('--parent_size',
                        type=int,
                        default=20,
                        help='Size of the parent genes.')

    parser.add_argument('--mutation_size',
                        type=int,
                        default=40,
                        help='Size of the mutated genes.')
    
    parser.add_argument('--mutation_prob',
                        type=float,
                        default=0.3,
                        help='Probability of mutation.')

    parser.add_argument('--crossover_size',
                        type=int,
                        default=40,
                        help='Size of the crossovered genes.')

    parser.add_argument('--n_iter',
                        type=int,
                        default=10,
                        help='Number of search iterations.')

    parser.add_argument('--n_layer_choice',
                        nargs='+',
                        type=int,
                        default=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                        help='Possible number of layers.')

    parser.add_argument('--d_model_choice',
                        nargs='+',
                        type=int,
                        default=[128, 256, 512, 650, 800],
                        help='Possible model dimensions.')

    parser.add_argument('--d_inner_choice',
                        nargs='+',
                        type=int,
                        default=list(range(512, 2049, 50))+list(range(2048, 3072, 200)),
                        help='Possible inner dimensions.')

    parser.add_argument('--n_head_choice',
                        nargs='+',
                        type=int,
                        default=[2, 4, 8],
                        help='Possible number of attention heads.')

    parser.add_argument('--param_constraint',
                        type=int,
                        default=5e6,
                        help='Number of parameters contraint.')

    parser.add_argument('--latency_scale',
                        type=float,
                        default=2.0,
                        help='How much latencies should be scaled.')

    parser.add_argument('--n_threads',
                        type=int,
                        default=1,
                        help='Number of inference threads.')

    parser.add_argument('--latency_repeat',
                        type=int,
                        default=5,
                        help='Number of latency measurements.')

    parser.add_argument('--pareto_search',
                        action='store_true',
                        help='Whether should conduct Pareto search ot not.')

    parser.add_argument('--device_name',
                        type=str,
                        default='XeonE5-2690',
                        help='Name of device that search is being conducted.')

    parser.add_argument('--eps',
                        type=float,
                        default=0.05,
                        help='Epsilon value.')

    parser.add_argument('--use_convex_hull',
                        action='store_true',
                        help='Whether should calculate convex hull or not.')

    parser.add_argument('--hybrid',
                        action='store_true',
                        help='Whether is a hybrid Pareto front or not.')

    parser.add_argument('--nsamples',
                        type=int,
                        default=20000,
                        help='Number of genes to be sampled.')

    parser.add_argument('--batch',
                        type=int,
                        default=1000,
                        help='Number of batched genes to conduct the brute force.')

    parser.add_argument('--do_train',
                        action='store_true',
                        help='Whether samples should be trained or not.')

    parser.add_argument('--start_train',
                        type=int,
                        default=40,
                        help='Search iteration that training should start.')

    parser.add_argument('--train_local',
                        action='store_true',
                        help='Whether samples should be locally trained or not.')

    parser.add_argument('--n_gpus',
                        type=int,
                        default=1,
                        help='Number of GPUs.')

    parser.add_argument('--gpu_config',
                        type=str,
                        default='dgx1_1gpu_fp32',
                        help='GPU configuration.')

    parser.add_argument('--config_file',
                        type=str,
                        default='wt103_base.yaml',
                        help='Configuration file.')

    parser.add_argument('--max_step',
                        type=int,
                        default=500,
                        help='Maximum number of training steps.')

    parser.add_argument('--experiment_name',
                        type=str,
                        default='evolution',
                        help='Name of the experiment.')

    parser.add_argument('--scheduler',
                        type=str,
                        default='constant',
                        help='Learning rate scheduler.')

    parser.add_argument('--use_valid',
                        action='store_true',
                        help='Whether validation set should be used or not.')

    parser.add_argument('--use_quantization',
                        action='store_true',
                        help='Whether quantization should be used or not.')

    parser.add_argument('--model_type',
                        type=str,
                        default='mem_transformer',
                        choices=['mem_transformer', 'hf_gpt2', 'hf_transfo_xl'],
                        help='Type of model to be searched.')

    parser.add_argument('--seed',
                        type=int,
                        default=1111,
                        help='Random seed.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # Gathers the command line arguments
    args = parse_args()

    # Applies random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Gathers the latency constraint based on device
    # TODO: this should get moved to config file with a command line override
    # NOTE: Making values large to make code run through faster 
    latency_constraint = {
        'XeonE5-2690': 10.0,
        'corei7': 10.0,
        'corei5': 10.0,
        'D3_V2': 10.0,
    }
    args.latency_constraint = latency_constraint[args.device_name]

    # Initializes the directory name
    path_to_amlt_results = './amlt_logs'
    dir_name = 'param_threshold_{}'.format(args.param_constraint / 1e6)
    
    # Adds missing strings to the directory name
    if args.pareto_search:
        dir_name += '_pareto'
    if args.use_convex_hull:
        dir_name += '_convex_hull'
    if args.start_train < args.n_iter:
        dir_name += '_wTrain'

    # TODO: the default path is ./evo_search which is inside the code repo!
    args.results_path = os.path.join(args.default_path, dir_name+'_'+args.device_name)

    # Standard evolutionary search
    if args.phase == 'run_search':
        results_dir = utils.full_path(args.results_path, create=True)
        with open(os.path.join(results_dir, 'search_config.yaml'), 'w') as f:
            yaml.dump(vars(args), f)

        run_search(vars(args), brute_force=False)

    # Submits ground-truth training jobs over the entire population after search
    elif args.phase == 'submit_gt_jobs':
        alg = Evolution(**vars(args))
        submit_ground_truth_jobs(vars(args),
                                 alg,
                                 max_step=40000,
                                 start_config=0,
                                 n_jobs=20,
                                 n_gpus=8,
                                 gpu_config='dgx1_8gpu_fp32',
                                 targets=['NLX-NDv2'])

    # Extracts proxy pareto from all samples seen during the evolutionary search
    elif args.phase == 'extract_pareto':
        ppl_eps = 1  # abosulte ppl difference for extracting the pareto
        param_eps = 0.01  # nomarlized parameter diff for extracting the pareto
        eps = ppl_eps if (args.start_train < args.n_iter and args.hybrid) else param_eps

        alg = Evolution(**vars(args))

        find_final_pareto_front(vars(args),
                                alg,
                                eps=eps,
                                hybrid=args.hybrid,
                                use_convex_hull=args.use_convex_hull)

    # Matches proxy pareto front points with the baseline and
    # submit selected points on the pareto front for full training
    elif args.phase == 'select_pareto':
        path_to_baseline = os.path.join(path_to_amlt_results, 'evolution_baselines')

        with open(os.path.join(path_to_baseline, 'params.pkl'), 'rb') as f:
            params_baseline = pickle.load(f)

        with open(os.path.join(path_to_baseline, 'latency_summary_{}.yaml'.format(args.device_name)), 'r') as f:
            latency_baseline = yaml.safe_load(f)

        baseline_params_list = []
        baseline_latency_list = []

        for k, p in params_baseline.items():
            baseline_params_list.append(p)
            baseline_latency_list.append(latency_baseline[k])

        fname = 'pareto{}'.format('' if args.hybrid else '_params')
        fname += '_convexHull' if args.use_convex_hull else ''

        path_to_logs = os.path.join(args.results_path, fname+'_points.pkl')
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
        plt.savefig(os.path.join(args.results_path, 'search_paretos{}.png'.format('' if args.hybrid else '_params')), bbox_inches='tight')

        path_to_logs = os.path.join(args.results_path, fname+'.pkl')
        with open(path_to_logs, 'rb') as f:
            is_pareto_dict = pickle.load(f)

        ks = list(is_pareto_dict.keys())
        for k in ks:
            if not k in keys_to_keep:
                is_pareto_dict[k] = False

        alg = Evolution(**vars(args))
        
        submit_pareto_front_jobs(vars(args),
                                 alg,
                                 is_pareto_dict,
                                 max_step=40000,
                                 start_config=0,
                                 bundle_count=20,
                                 n_gpus=8,
                                 gpu_config='dgx1_8gpu_fp32',
                                 targets=['NLX-NDv2'])

    # Compares baseline with searched Pareto results
    elif args.phase == 'compare_pareto':
        training_exp_name = 'pareto_{}'.format(args.device_name)

        alg = Evolution(**vars(args))

        compare_pareto_fronts(vars(args),
                              alg,
                              exp_name=training_exp_name,
                              path_to_dir=path_to_amlt_results,
                              start_config=0,
                              baseline_exp='evolution_baselines',
                              check_pareto=False)

    # Compares ground-truth Pareto with the proxy Pareto
    elif args.phase == 'gt_pareto':
        gt_exp_name = 'evolution_40000'
        os.makedirs(path_to_amlt_results, exist_ok=True)

        command = 'amlt results {} -I "*.json"  -o {}'.format(gt_exp_name, path_to_amlt_results)
        os.system(command)
        
        command = 'amlt results {} -I "*.yaml"  -o {}'.format(gt_exp_name, path_to_amlt_results)
        os.system(command)
        
        alg = Evolution(**vars(args))
        find_ground_truth_pareto(vars(args),
                                 alg,
                                 exp_name=gt_exp_name,
                                 path_to_dir=path_to_amlt_results,
                                 start_config=0,
                                 ppl_eps=0.1,
                                 latency_eps=0.01,
                                 hybrid=args.hybrid,
                                 use_convex_hull=args.use_convex_hull,
                                 min_acceptable_latency_diff=2,
                                 baseline_exp=None)

        alg = Evolution(**vars(args))
        compare_pareto_fronts(vars(args),
                              alg,
                              exp_name=gt_exp_name,
                              path_to_dir=path_to_amlt_results,
                              start_config=0,
                              baseline_exp='evolution_baselines',
                              check_pareto=True)

        # Compares final validation perplexity with nparams pareto
        with open('amlt_logs/evolution_40000/params_summary.yaml') as f:
            n_all_params = yaml.load(f)

        gt_results = parse_results_from_experiment('evolution_40000', 'amlt_logs/evolution_40000', file_type=['.json'], verbose=False)

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
        plt.savefig(os.path.join(args.results_path, fname), bbox_inches='tight')
