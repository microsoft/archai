# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

import torch
from transformers import AutoModelForCausalLM

from archai.common.file_utils import calculate_torch_model_size
from archai.quantization.ptq import dynamic_quantization_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-Training Quantization (PTQ) with a PyTorch model.")

    parser.add_argument("pre_trained_model_path", type=str, help="Path to the pre-trained model file.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    # Quantized model only uses maximum of 1 thread
    torch.set_num_threads(1)

    # Performs Post-Training Quantization (PTQ) over pre-trained model
    # Also loads the original pre-trained model for debugging
    model = AutoModelForCausalLM.from_pretrained(args.pre_trained_model_path)
    model_qnt = dynamic_quantization_torch(model)

    print(f"Model: {calculate_torch_model_size(model)}MB")
    print(f"Model-QNT: {calculate_torch_model_size(model_qnt)}MB")

    inputs = {"input_ids": torch.randint(1, 10, (1, 192))}
    logits = model(**inputs).logits
    logits_qnt = model_qnt(**inputs).logits

    print(f"Difference between logits: {logits_qnt - logits}")
