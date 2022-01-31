# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Validates the ONNX export by comparing outputs from PyTorch and ONNX.
"""

import argparse

import torch

from archai.nlp.compression.onnx.onnx_utils.onnx_loader import (load_from_onnx,
                                                                load_from_torch_for_export)


def parse_args():
    parser = argparse.ArgumentParser(description='Validates between PyTorch and exported ONNX model.')

    parser.add_argument('--torch_model_path',
                        type=str,
                        help='Path to the PyTorch model/checkpoint file.')

    parser.add_argument('--onnx_model_path',
                        type=str,
                        help='Path to the ONNX model file.')

    parser.add_argument('--model_type',
                        type=str,
                        default='mem_transformer',
                        choices=['mem_transformer', 'hf_gpt2', 'hf_transfo_xl'],
                        help='Type of model to be exported.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='Size of the batch.')

    parser.add_argument('--seq_len',
                        type=int,
                        default=32,
                        help='Sequence length.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # Gathers the command line arguments
    args = parse_args()

    # Transforms the command lines arguments into variables
    torch_model_path = args.torch_model_path
    onnx_model_path = args.onnx_model_path
    model_type = args.model_type
    batch_size = args.batch_size
    seq_len = args.seq_len

    # Loads PyTorch and ONNX models
    model, model_config = load_from_torch_for_export(model_type, torch_model_path)
    model_onnx = load_from_onnx(onnx_model_path)

    # Checks the type of attention to define the `past_key_values`
    n_past_values = 2
    if model_type == 'mem_transformer':
        if model_config['attn_type'] == 0:
            # `k`, `v` and relative embeddings
            n_past_values = 3

    # Defines PyTorch inputs
    torch.manual_seed(0)
    inputs = {
        'input_ids': torch.randint(0, model_config['n_token'], (batch_size, seq_len)),
        'past_key_values': tuple([
            torch.zeros(n_past_values, batch_size, model_config['n_head'], seq_len, model_config['d_head'])
            for _ in range(model_config['n_layer'])
        ])
    }

    # Defines ONNX inputs
    inputs_onnx = {'input_ids': inputs['input_ids'].cpu().detach().numpy()}
    for i in range(model_config['n_layer']):
        key = f'past_{i}'
        inputs_onnx[key] = inputs['past_key_values'][i].cpu().detach().numpy()

    # Performs the inference and compares the outputs
    probs = model(**inputs)[0].detach().numpy()
    probs_onnx = model_onnx.run(None, inputs_onnx)[0]

    print(f'Sum of differences between probs: {(probs_onnx - probs).sum()}')
