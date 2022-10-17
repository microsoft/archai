# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Extracts Pareto-frontier through Evolutionary Search, given constraints. 
"""

import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import yaml

from archai.common import utils
from archai.nlp.nas.evolution import Evolution
from archai.nlp.nas.nas_utils.constraints.constraint_pipeline import DEVICE_LATENCY_CONSTRAINT


def parse_args():
    parser = argparse.ArgumentParser(description='Language models Pareto-frontier extraction.')

    try:
        save_path = os.environ['AMLT_OUTPUT_DIR']
    except:
        save_path = '~/logdir' 

    search = parser.add_argument_group('Search configuration')
    search.add_argument('--default_path',
                        type=str,
                        default=save_path,
                        help='Path to the default folder used to save outputs.')

    search.add_argument('--model_type',
                        type=str,
                        default='mem_transformer',
                        choices=['hf_codegen', 'hf_gpt2', 'hf_gpt2_flex', 'hf_opt', 'hf_transfo_xl', 'mem_transformer'],
                        help='Type of model to be searched.')

    search.add_argument('--model_config',
                        type=str,
                        default=None,
                        help='YAML configuration file to override default configuration.')

    search.add_argument('--population_size',
                        type=int,
                        default=100,
                        help='Size of the population.')

    search.add_argument('--parent_size',
                        type=int,
                        default=20,
                        help='Size of the parent genes.')

    search.add_argument('--mutation_size',
                        type=int,
                        default=40,
                        help='Size of the mutated genes.')
    
    search.add_argument('--mutation_prob',
                        type=float,
                        default=0.3,
                        help='Probability of mutation.')

    search.add_argument('--crossover_size',
                        type=int,
                        default=40,
                        help='Size of the crossovered genes.')

    search.add_argument('--crossover_prob',
                        type=float,
                        default=0.5,
                        help='Probability of crossover.')

    search.add_argument('--n_iter',
                        type=int,
                        default=10,
                        help='Number of search iterations.')

    search.add_argument('--do_brute_force',
                        action='store_true',
                        help='Uses brute force instead of standard search.')

    search.add_argument('--n_samples',
                        type=int,
                        default=20000,
                        help='Number of genes used to sample during brute force.')

    search.add_argument('--batch',
                        type=int,
                        default=1000,
                        help='Number of batched genes used to conduct the brute force.')

    search.add_argument('--use_quantization',
                        action='store_true',
                        help='Uses quantized models to conduct the search.')

    search.add_argument('--seed',
                        type=int,
                        default=1111,
                        help='Random seed.')

    strategy = parser.add_argument_group('Training strategy')

    strategy.add_argument('--training_strategy',
                          type=str,
                          default='decoder_params',
                          choices=['decoder_params', 'val_ppl', 'char_accept_rate'],
                          help='Training strategy: decoder parameters, validation perplexity or character accept rate.')

    strategy.add_argument('--dataset',
                          type=str,
                          default='wt103',
                          choices=['wt103'],
                          help='Dataset (if not using `decoder_params`).')

    strategy.add_argument('--scoring_file',
                          type=str,
                          default=None,
                          help='Scoring .ljson file (if using `char_accept_rate`).')

    strategy.add_argument('--vocab_type',
                          type=str,
                          default='word',
                          choices=['word', 'bppe', 'gpt2'],
                          help='Type of vocabulary (if not using `decoder_params`).')

    strategy.add_argument('--vocab_size',
                          type=int,
                          default=10000,
                          help='Size of vocabulary (if not using `decoder_params`).')

    strategy.add_argument('--training_max_step',
                          type=int,
                          default=100,
                          help='Maximum number of training steps (if not using `decoder_params`).')

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

    constraint = parser.add_argument_group('Constraints')
    constraint.add_argument('--constraint_pipeline_type',
                            default='torch',
                            choices=['onnx', 'torch'],
                            help='Type of constraint pipeline to be used during search.')

    constraint.add_argument('--param_constraint_lower',
                            type=int,
                            default=5e6,
                            help='Candidates below total parameters will be rejected.')

    constraint.add_argument('--param_constraint_upper',
                            type=int,
                            default=12e6,
                            help='Candidates above total parameters will be rejected.')

    constraint.add_argument('--latency_constraint_upper',
                            type=float,
                            default=None,
                            help='Candidates above latency will be rejected.')

    constraint.add_argument('--n_threads',
                            type=int,
                            default=1,
                            help='Number of inference threads.')

    constraint.add_argument('--latency_repeat',
                            type=int,
                            default=10,
                            help='Number of latency measurements.')

    constraint.add_argument('--device_name',
                            type=str,
                            default='XeonE5-2690',
                            help='Name of device that search is being conducted on.')

    constraint.add_argument('--eps',
                            type=float,
                            default=0.05,
                            help='Value for neighborhood used around the Pareto front.')
                        
    args, _ = parser.parse_known_args()

    return vars(args)


if __name__ == '__main__':
    # Gathers the command line arguments
    args = parse_args()

    # Applies random seeds
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])

    if not torch.cuda.is_available():
        args['use_training_proxy'] = True
        print('No CUDA available, defaulting to `use_training_proxy` as True.')

    # Gathers the latency constraint based on device
    if args['latency_constraint_upper'] is None:
        args['latency_constraint_upper'] = DEVICE_LATENCY_CONSTRAINT[args['device_name']]

    # Initializes the result's path
    now = datetime.now()
    time_str = now.strftime('%d_%m_%Y_%H_%M_%S')
    results_path_str = f'{args["model_type"]}_lower_param_{args["param_constraint_lower"]/1e6}M_upper_param_{args["param_constraint_upper"]/1e6}M_latency_upper_{args["latency_constraint_upper"]}s_{args["device_name"]}_{time_str}'
    results_path = os.path.join(args['default_path'], results_path_str)
    args['results_path'] = utils.full_path(results_path, create=True)

    # Dumps the search configuration to a YAML file
    with open(os.path.join(args['results_path'], 'search_config.yaml'), 'w') as f:
        yaml.dump(args, f)

    # Loads model configuration file (if provided)
    try:
        with open(args['model_config'], 'r') as f:
            args['model_config'] = yaml.load(f, Loader=yaml.Loader)['train']
    except:
        args['model_config'] = {}

    # Creates the evolutionary search instance
    e = Evolution(args['results_path'],
                  model_type=args['model_type'],
                  model_config=args['model_config'],
                  population_size=args['population_size'],
                  parent_size=args['parent_size'],
                  mutation_size=args['mutation_size'],
                  mutation_prob=args['mutation_prob'],
                  crossover_size=args['crossover_size'],
                  crossover_prob=args['crossover_prob'],
                  n_iter=args['n_iter'],
                  use_quantization=args['use_quantization'],
                  training_strategy=args['training_strategy'],
                  dataset=args['dataset'],
                  scoring_file=args['scoring_file'],
                  vocab_type=args['vocab_type'],
                  vocab_size=args['vocab_size'],
                  training_max_step=args['training_max_step'],
                  constraint_pipeline_type=args['constraint_pipeline_type'],
                  param_constraint_lower=args['param_constraint_lower'],
                  param_constraint_upper=args['param_constraint_upper'],
                  latency_constraint_upper=args['latency_constraint_upper'],
                  n_threads=args['n_threads'],
                  latency_repeat=args['latency_repeat'],
                  n_layer=args['n_layer'],
                  d_model=args['d_model'],
                  d_inner=args['d_inner'],
                  n_head=args['n_head'])
    
    # Runs the evolutionary search or the brute force version
    e.run(do_brute_force=args['do_brute_force'],
          n_samples=args['n_samples'],
          batch=args['batch'])
