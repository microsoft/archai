# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import copy
import os
import random

import numpy as np
import torch
import yaml
from archai.common import utils
from archai.nlp.models.model_loader import load_config
from archai.nlp.nas.evolution import Evolution
from archai.nlp.nas.nas_utils.converter import Converter


def parse_args():
    parser = argparse.ArgumentParser(description='Samples random PyTorch architectures.')

    parser.add_argument('--n_samples',
                        type=int,
                        default=1000,
                        help='Number of architectures to sample')

    parser.add_argument('--n_layer',
                        nargs='+',
                        type=int,
                        default=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                        help='Possible number of layers.')

    parser.add_argument('--d_model',
                        nargs='+',
                        type=int,
                        default=[128, 256, 512, 784, 1024],
                        help='Possible model dimensions.')

    parser.add_argument('--d_inner',
                        nargs='+',
                        type=int,
                        default=list(range(512, 2049, 50))+list(range(2048, 3072, 200)),
                        help='Possible inner dimensions.')

    parser.add_argument('--n_head',
                        nargs='+',
                        type=int,
                        default=[8],
                        help='Possible number of attention heads.')

    parser.add_argument('--param_constraint_upper',
                        type=int,
                        default=1e10,
                        help='Any candidate above total parameters will be rejected.')

    parser.add_argument('--param_constraint_lower',
                        type=int,
                        default=1e5,
                        help='Any candidate below total parameters will be rejected.')

    parser.add_argument('--model_type',
                        type=str,
                        default='mem_transformer',
                        choices=['hf_gpt2', 'hf_gpt2_flex', 'hf_transfo_xl', 'mem_transformer'],
                        help='Type of model to be sampled.')

    parser.add_argument('--seed',
                        type=int,
                        default=1111,
                        help='Random seed.')

    parser.add_argument('--use_quantization',
                        action='store_true',
                        help='Whether quantization should be used or not.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # Gathers the command line arguments
    args = vars(parse_args())
    args['model_config'] = {}
    
    # Applies random seeds
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    
    # Creates a dictionary of choices
    choices = {
        'n_layer': args['n_layer'],
        'd_model': args['d_model'],
        'd_inner': args['d_inner'],
        'n_head': args['n_head']
    }

    # Initializes the model's search configuration
    model_config_search = load_config(args['model_type'], config_type='search')

    # Prevents non-available keys from being used during search
    # Also, overrides default search choices with inputted ones
    for k, v in choices.items():
        if k in model_config_search.keys() and v is not None:
            model_config_search[k]['value'] = v
    
    # Initializes the converter
    c = Converter(**model_config_search)

    # Initializes the evolutionary algorithm
    output_path = utils.full_path(f'sampled/{args["model_type"]}', create=True)
    e = Evolution(output_path, **args)

    print('Sampling genes ...')

    # Samples a set of architectures
    arch_genes = e.sample_random_population(args['n_samples'])

    for i, arch_gene in enumerate(arch_genes):
        # Gathers the default configuration based on model's type
        config_default = load_config(args['model_type'], config_type='default')
        
        # Converts gene to config
        arch_config = c.gene_to_config(arch_gene)

        # Updates the configuration
        model_config = copy.deepcopy(config_default)
        model_config.update(arch_config)

        # Updates somes arguments to prevent future ONNX export from failing
        model_config['div_val'] = 1
        model_config['d_head'] = -1
        model_config['n_token'] = 10000
        model_config['cutoffs'] = []
        model_config['tie_projs'] = [False]

        # Dumps the configuration for future reference
        with open(os.path.join(output_path, f'config_{i}.yaml'), 'w') as f:
            yaml.dump(model_config, f)
