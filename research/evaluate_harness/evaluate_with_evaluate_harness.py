# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json

from evaluate_harness.suites.glue import GLUESuite


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluates pre-trained models using `evaluate`.")

    parser.add_argument("pre_trained_model_path", type=str, help="Path to the pre-trained model file.")

    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=None,
        help="Path to the saved outputs.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    suite = GLUESuite()
    outputs = suite.run(args.pre_trained_model_path)

    output_json = json.dumps(outputs, indent=2)
    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(output_json)

    print(outputs)
