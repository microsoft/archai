# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from archai.nlp.trainers.hf.training_args import DistillerTrainingArguments


def test_distiller_training_arguments():
    # Assert that the default values for alpha and temperature are correct
    args = DistillerTrainingArguments("tmp")
    assert args.alpha == 0.5
    assert args.temperature == 1.0

    # Assert that the custom values for alpha and temperature are correct
    args = DistillerTrainingArguments("tmp", alpha=0.75, temperature=1.25)
    assert args.alpha == 0.75
    assert args.temperature == 1.25
