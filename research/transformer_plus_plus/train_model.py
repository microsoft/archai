import sys
from pathlib import Path

self_dir = Path(__file__).absolute().parent
sys.path.append(str(self_dir.parent))

import argparse
import os

from transformer_plus_plus.training.experiment import Experiment

from archai.common.file_utils import check_available_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs Transformer++ experiment.")

    parser.add_argument(
        "arch_config_file",
        type=str,
        help="Path to the architecture configuration file.",
    )

    parser.add_argument(
        "experiment_config_file",
        type=str,
        help="Path to the experiment configuration file.",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="",
        help="Defines an output folder for the saved outputs.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    experiment = Experiment(args.arch_config_file, args.experiment_config_file, output_dir=args.output_dir)

    # Asserts that resume_from_checkpoint will be a valid boolean
    # when pre-trained checkpoints exists
    results_output_dir = os.path.join(args.output_dir, experiment.results_dir)
    resume_from_checkpoint = check_available_checkpoint(results_output_dir)

    experiment.run(resume_from_checkpoint=resume_from_checkpoint)
