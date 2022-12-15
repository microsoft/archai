# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import shutil
import tempfile

import torch

from archai.nlp.file_utils import (
    calculate_onnx_model_size,
    calculate_torch_model_size,
    check_available_checkpoint,
    create_file_name_identifier,
)


def test_calculate_onnx_model_size():
    with tempfile.TemporaryFile() as tmp:
        tmp.write(b"a" * 10**6)
        tmp.seek(0)

        # Assert that the calculated size is correct
        size = calculate_onnx_model_size(tmp.name)
        assert size == 1.0


def test_calculate_torch_model_size():
    model = torch.nn.Linear(10, 5)

    # Assert that the calculated size is correct
    size = calculate_torch_model_size(model)
    assert size > 0


def test_check_available_checkpoint():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Assert that the folder is empty
        assert not check_available_checkpoint(tmp_dir)

        # Assert that the folder is not empty
        subfolder = os.path.join(tmp_dir, "checkpoint-1")
        os.mkdir(subfolder)
        assert check_available_checkpoint(tmp_dir)

        # Assert that the folder is empty again
        shutil.rmtree(subfolder)
        assert not check_available_checkpoint(tmp_dir)


def test_create_file_name_identifier():
    # Assert with a few different file names and identifiers
    assert create_file_name_identifier("file.txt", "-abc") == "file-abc.txt"
    assert create_file_name_identifier("/path/to/file.txt", "-abc") == "/path/to/file-abc.txt"
    assert create_file_name_identifier("/path/to/file.txt", "-123") == "/path/to/file-123.txt"
