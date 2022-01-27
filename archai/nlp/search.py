# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Performs evolutionary search to find optimal architectures.
"""

import argparse
import os
import random

import numpy as np
import torch
import yaml

from archai.common import utils
from archai.nlp.nas.evolution import run_search


def parse_args():
    parser = argparse.ArgumentParser(description='Evolutionary search for autoregressive language models.')

    parser.add_argument('--default_path',
                        type=str,
                        default='./evo_search',
                        help='Path to the folder that will save the results.')

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
                        default=[128, 256, 512, 768, 1024],
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

    parser.add_argument('--param_constraint_upper',
                        type=int,
                        default=12e6,
                        help='Any candidate above total parameters will be rejected.')

    parser.add_argument('--param_constraint_lower',
                        type=int,
                        default=5e6,
                        help='Any candidate below total parameters will be rejected.')

    parser.add_argument('--n_threads',
                        type=int,
                        default=1,
                        help='Number of inference threads.')

    parser.add_argument('--latency_repeat',
                        type=int,
                        default=5,
                        help='Number of latency measurements.')

    parser.add_argument('--device_name',
                        type=str,
                        default='XeonE5-2690',
                        help='Name of device that search is being conducted.')

    parser.add_argument('--eps',
                        type=float,
                        default=0.05,
                        help='Epsilon value.')

    parser.add_argument('--n_samples',
                        type=int,
                        default=20000,
                        help='Number of genes to be sampled.')

    parser.add_argument('--batch',
                        type=int,
                        default=1000,
                        help='Number of batched genes to conduct the brute force.')

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
    args.latency_constraint_upper = latency_constraint[args.device_name]

    # Initializes the directory name
    path_to_amlt_results = './amlt_logs'
    dir_name = f'lower_param_thresh_{args.param_constraint_lower/1e6}_upper_param_thresh_{args.param_constraint_upper/1e6}_latency_upper_thresh_{args.latency_constraint_upper/1e6}'
    
    
    # TODO: the default path is ./evo_search which is inside the code repo!
    args.results_path = os.path.join(args.default_path, dir_name+'_'+args.device_name)

    # Standard evolutionary search
    results_dir = utils.full_path(args.results_path, create=True)
    with open(os.path.join(results_dir, 'search_config.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    run_search(vars(args), brute_force=False)
    
    
    
