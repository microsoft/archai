# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

import numpy as np
import torch

from archai.onnx.onnx_loader import load_from_onnx


def parse_args():
    parser = argparse.ArgumentParser(description="Validates past key/values with an ONNX model.")

    parser.add_argument("onnx_model_path", type=str, help="Path to the ONNX model file.")

    parser.add_argument("-nh", "--n_head", type=int, default=12, help="Number of attention heads.")

    parser.add_argument("-dh", "--d_head", type=int, default=64, help="Dimension of attention head.")

    parser.add_argument("-bs", "--batch_size", type=int, default=1, help="Size of the batch.")

    parser.add_argument("-sl", "--seq_len", type=int, default=32, help="Sequence length.")

    parser.add_argument("-psl", "--past_seq_len", type=int, default=32, help="Past key/values sequence length.")

    parser.add_argument("-npv", "--n_past_values", type=int, default=2, help="Number of past key/values.")

    parser.add_argument("-nl", "--n_layers", type=int, default=12, help="Number of layers.")

    parser.add_argument("-nt", "--n_tokens", type=int, default=10000, help="Number of tokens for sampling.")

    parser.add_argument("-nr", "--n_runs", type=int, default=100, help="Number of comparisons.")

    parser.add_argument("-nti", "--new_token_id", type=int, default=6, help="Identifier of token to be predicted.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    accuracy = 0.0
    for i in range(args.n_runs):
        torch.manual_seed(i)
        np.random.seed(i)

        model_onnx = load_from_onnx(args.onnx_model_path)

        inputs = {"input_ids": np.random.randint(0, args.n_tokens, (args.batch_size, args.seq_len), dtype=np.int64)}
        for i in range(args.n_layers):
            key = f"past_{i}"
            inputs[key] = np.zeros(
                (args.n_past_values, args.batch_size, args.n_head, args.past_seq_len, args.d_head), dtype=np.float32
            )

        # 1st inference (full pass with initial inputs)
        outputs = model_onnx.run(None, inputs)

        # 2nd inference (partial pass with only `new_token_id`)
        inputs_p = {"input_ids": np.array([[args.new_token_id]], dtype=np.int64)}
        for i in range(args.n_layers):
            key = f"past_{i}"
            inputs_p[key] = outputs[i + 1]
        outputs_partial = model_onnx.run(None, inputs_p)

        # 3rd inference (full pass with initial inputs and `new_token_id`)
        inputs["input_ids"] = np.expand_dims(np.append(inputs["input_ids"], args.new_token_id), 0)
        outputs_full = model_onnx.run(None, inputs)

        accuracy += np.argmax(outputs_partial[0]) == np.argmax(outputs_full[0])

        print(f"Partial pass (with past key/value) sample token: {np.argmax(outputs_partial[0])}")
        print(f"Full pass (without past key/value) sample token: {np.argmax(outputs_full[0])}")

    accuracy /= args.n_runs
    print(f"Acc: {accuracy}")
