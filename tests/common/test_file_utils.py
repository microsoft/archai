# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import shutil
import tempfile
import torch

from archai.common.file_utils import (
    calculate_onnx_model_size,
    calculate_torch_model_size,
    check_available_checkpoint,
    copy_file,
    create_empty_file,
    create_file_name_identifier,
    create_file_with_string,
    get_full_path,
    TemporaryFiles,
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


def test_create_empty_file():
    # Assert that the file is created and is empty
    file_path = "empty_file.txt"
    create_empty_file(file_path)

    assert os.path.exists(file_path)
    assert os.path.getsize(file_path) == 0

    os.remove(file_path)


def test_create_file_with_string():
    # Assert that the file is created and contains the string
    file_path = "file_with_string.txt"
    content = "Hello World!"

    create_file_with_string(file_path, content)
    assert os.path.exists(file_path)
    assert os.path.getsize(file_path) > 0

    with open(file_path, "r") as f:
        assert f.read() == content

    os.remove(file_path)


def test_copy_file():
    # Assert that the file is copied correctly
    src_file_path = "src_file.txt"
    dest_file_path = "dest_file.txt"

    content = "Hello World!"
    with open(src_file_path, "w") as f:
        f.write(content)

    copy_file(src_file_path, dest_file_path)

    assert os.path.exists(dest_file_path)
    assert os.path.getsize(dest_file_path) > 0

    with open(dest_file_path, "r") as f:
        assert f.read() == content

    os.remove(src_file_path)
    os.remove(dest_file_path)


def test_get_full_path():
    # Assert that the path is correct
    path = "~/example_folder"
    full_path = get_full_path(path, create_folder=True)

    assert os.path.exists(full_path)

    os.rmdir(full_path)


def test_temporary_files():
    with TemporaryFiles() as tmp:
        name1 = tmp.get_temp_file()
        name2 = tmp.get_temp_file()
        with open(name1, 'w') as f:
            f.write("test1")
        with open(name2, 'w') as f:
            f.write("test2")
        with open(name1, 'r') as f:
            assert f.readline() == 'test1'
        with open(name2, 'r') as f:
            assert f.readline() == 'test2'

    assert not os.path.isfile(name1)
    assert not os.path.isfile(name2)
