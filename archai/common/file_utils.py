# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pathlib
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from types import TracebackType

import torch

# File-related constants
CHECKPOINT_FOLDER_PREFIX = "checkpoint"
CHECKPOINT_REGEX = re.compile(r"^" + CHECKPOINT_FOLDER_PREFIX + r"\-(\d+)$")


def calculate_onnx_model_size(model_path: str) -> float:
    """Calculate the size of an ONNX model.

    This function calculates the size of an ONNX model by reading the size of
    the file on disk.

    Args:
        model_path: The path to the ONNX model on disk.

    Returns:
        The size of the model in megabytes.

    """

    size = os.path.getsize(model_path) / 1e6

    return size


def calculate_torch_model_size(model: torch.nn.Module) -> float:
    """Calculate the size of a PyTorch model.

    This function calculates the size of a PyTorch model by saving its state
    dictionary to a temporary file and reading the size of the file on disk.

    Args:
        model: The PyTorch model.

    Returns:
        The size of the model in megabytes.

    """

    torch.save(model.state_dict(), "temp.p")

    size = os.path.getsize("temp.p") / 1e6

    os.remove("temp.p")

    return size


def check_available_checkpoint(folder_name: str) -> bool:
    """Check if there are any available checkpoints in a given folder.

    This function checks if a given folder contains any checkpoints by looking
    for directories that match a regular expression for checkpoint names.

    Args:
        folder_name: The path to the folder that might contain checkpoints.

    Returns:
        `True` if there are available checkpoints, `False` otherwise.

    """

    if not os.path.exists(folder_name):
        return False

    folder_content = os.listdir(folder_name)
    checkpoints = [
        path
        for path in folder_content
        if CHECKPOINT_REGEX.search(path) is not None and os.path.isdir(os.path.join(folder_name, path))
    ]

    if len(checkpoints) == 0:
        return False

    return True


def create_file_name_identifier(file_name: str, identifier: str) -> str:
    """Create a new file name by adding an identifier to the end
    of an existing file name (before the file extension).

    Args:
        file_name: The original file name.
        identifier: The identifier to be added to the file name.

    Returns:
        The new file name with the added identifier.

    """

    file_name = Path(file_name)
    file_name_identifier = file_name.parent.joinpath(file_name.stem + identifier).with_suffix(file_name.suffix)

    return file_name_identifier.as_posix()


def create_empty_file(file_path: str) -> None:
    """Create an empty file at the given path.

    Args:
        file_path: The path to the file to be created.

    """

    open(file_path, "w").close()


def create_file_with_string(file_path: str, content: str) -> None:
    """Create a file at the given path and writes the given string to it.

    Args:
        file_path: The path to the file to be created.
        content: The string to be written to the file.

    """

    pathlib.Path(file_path).write_text(content)


def copy_file(
    src_file_path: str, dest_file_path: str, force_shutil: Optional[bool] = True, keep_metadata: Optional[bool] = False
) -> str:
    """Copy a file from one location to another.

    Args:
        src_file_path: The path to the source file.
        dest_file_path: The path to the destination file.
        force_shutil: Whether to use `shutil` to copy the file.
        keep_metadata: Whether to keep source file metadata when copying.

    Returns:
        The path to the destination file.

    """

    def _copy_file_basic_mode(src_file_path: str, dest_file_path: str) -> str:
        if os.path.isdir(dest_file_path):
            dest_file_path = os.path.join(dest_file_path, pathlib.Path(src_file_path).name)

        with open(src_file_path, "rb") as src, open(dest_file_path, "wb") as dest:
            dest.write(src.read())

        return dest_file_path

    if not force_shutil:
        return _copy_file_basic_mode(src_file_path, dest_file_path)

    # Note shutil.copy2 might fail on Azure if file system does not support OS level copystats
    # Use keep_metadata=True only if needed for maximum compatibility
    try:
        copy_fn = shutil.copy2 if keep_metadata else shutil.copy
        return copy_fn(src_file_path, dest_file_path)
    except OSError as e:
        if keep_metadata or e.errno != 38:  # OSError 38: Function not implemented
            raise

        return _copy_file_basic_mode(src_file_path, dest_file_path)


def get_full_path(path: str, create_folder: Optional[bool] = False) -> str:
    """Get the full path to a file or folder.

    Args:
        path: The path to the file or folder.
        create_folder: Whether to create the folder if it does not exist.

    Returns:
        The full path to the file or folder.

    """

    assert path

    path = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))

    if create_folder:
        os.makedirs(path, exist_ok=True)

    return path


class TemporaryFiles:
    """ Windows has a weird quirk where the tempfile.NamedTemporaryFile cannot be opened a second time. """
    def __init__(self):
        self.files_to_delete = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type: type[BaseException], exc_val: BaseException, exc_tb: TracebackType) -> None:
        for name in self.files_to_delete:
            os.unlink(name)
        self.files_to_delete = []

    def get_temp_file(self) -> str:
        result = None
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            result = tmp.name
        self.files_to_delete += [result]
        return result
