# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Validates the usage of past key/values with an ONNX model.
"""

import argparse

import numpy as np
import torch

from archai.nlp.compression.onnx.onnx_utils.onnx_loader import load_from_onnx


def parse_args():
    parser = argparse.ArgumentParser(description='Validates between PyTorch and exported ONNX model.')

    parser.add_argument('--onnx_model_path',
                        type=str,
                        help='Path to the ONNX model file.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='Size of the batch.')

    parser.add_argument('--seq_len',
                        type=int,
                        default=32,
                        help='Sequence length.')

    parser.add_argument('--past_seq_len',
                        type=int,
                        default=32,
                        help='Past key/values sequence length.')

    parser.add_argument('--n_runs',
                        type=int,
                        default=100,
                        help='Number of comparisons.')

    parser.add_argument('--new_token_id',
                        type=int,
                        default=6,
                        help='Identifier of token to be predicted.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # Gathers the command line arguments
    args = parse_args()

    # Transforms the command lines arguments into variables
    onnx_model_path = args.onnx_model_path
    batch_size = args.batch_size
    seq_len = args.seq_len
    past_seq_len = args.past_seq_len
    n_runs = args.n_runs
    new_token_id = args.new_token_id

    # Initializes accuracy as 0
    accuracy = 0.0

    for i in range(n_runs):
        # Sets torch and numpy seeds
        torch.manual_seed(i)
        np.random.seed(i)

        # Loads ONNX model
        model_onnx = load_from_onnx(onnx_model_path)

        # Defines random inputs and zero-valued past states
        inputs = {'input_ids': np.random.randint(0, 10000, (batch_size, seq_len), dtype=np.int64)}
        for i in range(2):
            key = f'past_{i}'
            inputs[key] = np.zeros((2, batch_size, 8, 0, past_seq_len), dtype=np.float32)

        # 1st inference (full pass with initial inputs)
        probs, past_key_values_0, past_key_values_1 = model_onnx.run(None, inputs)

        # 2nd inference (partial pass with only `new_token_id`)
        inputs_p = {'input_ids': np.array([[new_token_id]], dtype=np.int64)}
        inputs_p['past_0'] = past_key_values_0
        inputs_p['past_1'] = past_key_values_1
        probs_partial, past_key_values_0, past_key_values_1 = model_onnx.run(None, inputs_p)

        # 3rd inference (full pass with initial inputs and `new_token_id`)
        inputs['input_ids'] = np.expand_dims(np.append(inputs['input_ids'], new_token_id), 0)
        probs_full, _, _ = model_onnx.run(None, inputs)

        # Calculates the accuracy
        accuracy += np.argmax(probs_partial) == np.argmax(probs_full)

        # print(f'Partial pass (with past key/value) sample token: {np.argmax(probs_partial)}')
        # print(f'Full pass (without past key/value) sample token: {np.argmax(probs_full)}')

    accuracy /= n_runs
    print(f'Acc: {accuracy}')
    