# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile

import pytest

from archai.nlp.datasets.nvidia.corpus_utils import create_dirs, get_dataset_dir_name


def test_get_dataset_dir_name():
    # Assert that the correct dataset directory name is returned for supported datasets
    assert get_dataset_dir_name("wt2") == "wikitext-2"
    assert get_dataset_dir_name("wt103") == "wikitext-103"
    assert get_dataset_dir_name("lm1b") == "one-billion-words"
    assert get_dataset_dir_name("olx_jobs") == "olx_jobs"

    # Assert that a RuntimeError is raised for unsupported datasets
    with pytest.raises(RuntimeError):
        get_dataset_dir_name("unsupported_dataset")


def test_create_dirs():
    experiment_name = "experiment"
    output_dir = tempfile.mkdtemp()
    dataroot = tempfile.mkdtemp()
    dataset_name = "wt2"

    # Assert that the function creates the expected directories
    dataset_dir, output_dir, pretrained_path, cache_dir = create_dirs(
        dataroot, dataset_name, experiment_name=experiment_name, output_dir=output_dir
    )
    assert os.path.isdir(dataset_dir)
    assert os.path.isdir(output_dir)
    assert os.path.isdir(cache_dir)
    assert pretrained_path == ""
