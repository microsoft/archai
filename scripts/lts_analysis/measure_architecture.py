# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import csv
import glob
import os

import yaml
from archai.nlp.models.model_loader import load_model_from_config
from archai.nlp.nas.nas_utils.constraints.constraint_pipeline import TorchConstraintPipeline

# Constants
SEQ_LEN = 64


def parse_args():
    parser = argparse.ArgumentParser(description='Measures PyTorch architectures.')

    parser.add_argument('--input_path',
                        type=str,
                        default=None,
                        help='Path to the input folder that holds configuration files.')

    parser.add_argument('--model_type',
                        type=str,
                        default='mem_transformer',
                        choices=['hf_gpt2', 'hf_gpt2_flex', 'hf_transfo_xl', 'mem_transformer'],
                        help='Type of model to be sampled.')

    parser.add_argument('--use_quantization',
                        action='store_true',
                        help='Whether quantization should be used or not.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # Gathers the command line arguments
    args = vars(parse_args())
    
    if args['use_quantization']:
        is_quantized = '_qnt'
    else:
        is_quantized = ''
    
    # Dumps header to a .csv file
    output_csv_path = os.path.join(args['input_path'], f'torch{is_quantized}.csv')
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['idx', 'params', 'total_params', f'latency{is_quantized}', f'memory{is_quantized}'])

    # Defines the measurement pipeline
    pipeline = TorchConstraintPipeline(use_quantization=args['use_quantization'], seq_len=SEQ_LEN)

    # Finds all configuration files and measures their architectures
    config_file_paths = glob.glob(os.path.join(args['input_path'], '*.yaml'))
    for i, config_file in enumerate(config_file_paths):
        print(f'Architecture: {i+1}/{len(config_file_paths)}')

        # Loads the YAML file
        with open(config_file, 'r') as f:
            config = yaml.load(f, yaml.Loader)

        # Creates the model
        model = load_model_from_config(args['model_type'], config)

        # Performs the measurement
        params, total_params, latency, memory = pipeline(model)

        # Dumps results to a .csv file
        with open(output_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i, params, total_params, latency, memory])
