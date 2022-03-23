# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Profiles (characterizes) language model search spaces, which is helpful 
    for gaining intuition into the nature of the search space.
"""

import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import yaml

from archai.common import utils
from archai.nlp.nas.nas_utils.search_space_profiler import SearchSpaceProfiler


def parse_args():
    parser = argparse.ArgumentParser(description='Profiles the search space of language models.')

    try:
        save_path = os.environ['AMLT_OUTPUT_DIR']
    except:
        save_path = '~/logdir' 

    profiler = parser.add_argument_group('Profiler configuration')
    profiler.add_argument('--default_path',
                          type=str,
                          default=save_path,
                          help='Path to the default folder used to save outputs.')

    profiler.add_argument('--model_type',
                          type=str,
                          default='hf_gpt2_flex',
                          choices=['hf_gpt2', 'hf_gpt2_flex', 'hf_transfo_xl', 'mem_transformer'],
                          help='Type of model to be profiled.')

    profiler.add_argument('--model_config',
                          type=str,
                          default=None,
                          help='YAML configuration file to override default configuration.')

    profiler.add_argument('--use_quantization',
                          action='store_true',
                          help='Uses quantized models to measure latency and memory.')

    profiler.add_argument('--seed',
                          type=int,
                          default=1111,
                          help='Random seed.')

    profiler.add_argument('--constraint_pipeline_type',
                          default='torch',
                          choices=['torch'],
                          help='Type of pipeline to be used during profiling.')

    profiler.add_argument('--n_threads',
                          type=int,
                          default=1,
                          help='Number of inference threads.')

    profiler.add_argument('--latency_repeat',
                          type=int,
                          default=5,
                          help='Number of latency measurements.')

    choice = parser.add_argument_group('Hyperparameters choices')
    choice.add_argument('--n_layer',
                        nargs='+',
                        type=int,
                        default=None,
                        help='Choices for number of layers.')

    choice.add_argument('--d_model',
                        nargs='+',
                        type=int,
                        default=None,
                        help='Choices for model dimensions.')

    choice.add_argument('--d_inner',
                        nargs='+',
                        type=int,
                        default=None,
                        help='Choices for inner dimensions.')

    choice.add_argument('--n_head',
                        nargs='+',
                        type=int,
                        default=None,
                        help='Choices for number of attention heads.')
                        
    args, _ = parser.parse_known_args()

    return vars(args)


if __name__ == '__main__':
    # Gathers the command line arguments
    args = parse_args()

    # Applies random seeds
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])

    # Initializes the result's path
    now = datetime.now()
    time_str = now.strftime("%d_%m_%Y_%H_%M_%S")
    results_path_str = f'profile_{args["model_type"]}_{time_str}'
    results_path = os.path.join(args['default_path'], results_path_str)
    args['results_path'] = utils.full_path(results_path, create=True)

    # Dumps the search configuration to a YAML file
    with open(os.path.join(args['results_path'], 'profiler_config.yaml'), 'w') as f:
        yaml.dump(args, f)

    # Loads model configuration file (if provided)
    try:
        with open(args['model_config'], 'r') as f:
            args['model_config'] = yaml.load(f, Loader=yaml.Loader)['train']
    except:
        args['model_config'] = {}

    # Profiles the search space
    profiler = SearchSpaceProfiler(args['results_path'],
                                   model_type=args['model_type'],
                                   model_config=args['model_config'],
                                   use_quantization=args['use_quantization'],
                                   constraint_pipeline_type=args['constraint_pipeline_type'],
                                   n_threads=args['n_threads'],
                                   latency_repeat=args['latency_repeat'],
                                   n_layer=args['n_layer'],
                                   d_model=args['d_model'],
                                   d_inner=args['d_inner'],
                                   n_head=args['n_head'])
    profiler.run()
