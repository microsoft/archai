# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os

from experiment_runner import ExperimentRunner

from archai.nlp.file_utils import check_available_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs an experiment.")

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

    args, extra_args = parser.parse_known_args()

    return args, extra_args


if __name__ == "__main__":
    args, extra_args = parse_args()

    experiment = ExperimentRunner(args.experiment_config_file, 
                                output_dir=args.output_dir,
                                cmdline_args=extra_args)

    # Asserts that resume_from_checkpoint will be a valid boolean
    # when pre-trained checkpoints exists
    results_output_dir = os.path.join(args.output_dir, experiment.results_dir)
    resume_from_checkpoint = check_available_checkpoint(results_output_dir)

    experiment.run(resume_from_checkpoint=resume_from_checkpoint)
