# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""File-related utilities."""

import os
import re
from pathlib import Path

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
