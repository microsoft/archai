# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Exports PyTorch-based saved models to ONNX.
"""

import argparse

from archai.nlp.compression.onnx.onnx_utils.export import export_onnx_from_torch
from archai.nlp.compression.onnx.onnx_utils.onnx_loader import load_from_torch_for_export
from archai.nlp.compression.onnx.onnx_utils.optimization import optimize_onnx
from archai.nlp.compression.quantization.ptq import dynamic_quantization_onnx


def parse_args():
    parser = argparse.ArgumentParser(description='Exports a PyTorch-based model to ONNX.')

    parser.add_argument('--torch_model_path',
                        type=str,
                        help='Path to the PyTorch model/checkpoint file.')

    parser.add_argument('--onnx_model_path',
                        type=str,
                        help='Path to the output ONNX model file.')

    parser.add_argument('--model_type',
                        type=str,
                        default='mem_transformer',
                        choices=['mem_transformer', 'hf_gpt2', 'hf_gpt2_flex', 'hf_transfo_xl'],
                        help='Type of model to be exported.')

    parser.add_argument('--opset_version',
                        type=int,
                        default=11,
                        help='Version of ONNX operators.')

    parser.add_argument('--opt_level',
                        type=int,
                        default=0,
                        help='Level of the ORT optimization.')

    parser.add_argument('--num_heads',
                        type=int,
                        default=8,
                        help='Number of attention heads (for fusion).')

    parser.add_argument('--optimization',
                        action='store_true',
                        help='Applies optimization to the exported model.')

    parser.add_argument('--quantization',
                        action='store_true',
                        help='Applies dynamic quantization to the exported model.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # Gathers the command line arguments
    args = parse_args()

    # Transforms the command lines arguments into variables
    torch_model_path = args.torch_model_path
    onnx_model_path = args.onnx_model_path
    model_type = args.model_type
    opset_version = args.opset_version
    opt_level = args.opt_level
    num_heads = args.num_heads
    optimization = args.optimization
    quantization = args.quantization

    # Loads the PyTorch model
    model, model_config = load_from_torch_for_export(model_type, torch_model_path)

    # Exports to ONNX
    export_onnx_from_torch(model,
                           model_config,
                           model_type,
                           onnx_model_path,
                           share_weights=True,
                           opset_version=opset_version)

    # Whether optimization should be applied
    if optimization:
        ort_model_path = optimize_onnx(model_type,
                                       onnx_model_path,
                                       num_heads=num_heads,
                                       opt_level=opt_level)

        # Caveat to enable quantization after optimization
        onnx_model_path = ort_model_path

    # Whether dynamic quantization should be applied
    if quantization:
        qnt_model_path = dynamic_quantization_onnx(onnx_model_path)
