# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import os

from transformers import GPT2LMHeadModel

from archai.nlp.file_utils import calculate_onnx_model_size
from archai.nlp.onnx import export_to_onnx, optimize_onnx
from archai.nlp.quantization import dynamic_quantization_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exports a GPT-2 model to ONNX.")

    parser.add_argument("output_model_path", type=str, help="Path to the ONNX output model.")

    parser.add_argument("-op", "--opset", type=int, default=11, help="ONNX opset version.")

    parser.add_argument(
        "-a",
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute difference to be tolerated between input and output models.",
    )

    parser.add_argument("-ol", "--opt_level", type=int, default=1, help="Level of the ORT optimization.")

    parser.add_argument("-opt", "--optimization", action="store_true", help="Optimizes the exported model.")

    parser.add_argument(
        "-qnt",
        "--quantization",
        action="store_true",
        help="Dynamically quantizes the exported model.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    onnx_config = export_to_onnx(
        model,
        args.output_model_path,
        task="causal-lm",
        use_past=True,
        share_weights=True,
        opset=args.opset,
        atol=args.atol,
    )
    print(f"Model: {calculate_onnx_model_size(args.output_model_path)}MB")

    if args.optimization:
        ort_model_path = optimize_onnx(args.output_model_path, onnx_config, opt_level=args.opt_level)
        args.output_model_path = ort_model_path
        print(f"Model-OPT: {calculate_onnx_model_size(args.output_model_path)}MB")

    if args.quantization:
        qnt_model_path = dynamic_quantization_onnx(args.output_model_path)
        print(f"Model-QNT: {calculate_onnx_model_size(qnt_model_path)}MB")

    # Exports model's configuration to JSON
    model_config_path = os.path.join(os.path.dirname(args.output_model_path), "config.json")
    with open(model_config_path, "w") as f:
        json.dump(onnx_config.config.to_dict(), f)
