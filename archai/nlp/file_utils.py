# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""File-related utilities.
"""

import os
import re
from pathlib import Path

import torch

# File-related constants
CHECKPOINT_FOLDER_PREFIX = "checkpoint"
CHECKPOINT_REGEX = re.compile(r"^" + CHECKPOINT_FOLDER_PREFIX + r"\-(\d+)$")


def calculate_onnx_model_size(model_path: str) -> float:
    """Calculates an ONNX model size.

    Args:
        model_path: Path of the ONNX model.

    Returns:
        (float): Size of the model (MB).

    """

    size = os.path.getsize(model_path) / 1e6

    return size


def calculate_torch_model_size(model: torch.nn.Module) -> float:
    """Calculates a PyTorch model size.

    Args:
        model: PyTorch model.

    Returns:
        (float): Size of the model (MB).

    """

    torch.save(model.state_dict(), "temp.p")

    size = os.path.getsize("temp.p") / 1e6

    os.remove("temp.p")

    return size


def check_available_checkpoint(folder_name: str) -> bool:
    """Checks if there are any available checkpoints.

    Args:
        folder_name: Path to the folder that might contain checkpoints.

    Returns:
        (bool): Whether there are available checkpoints.

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
    """Adds an identifier (suffix) to the end of the file name.

    Args:
        file_name: Path to have a suffix added.
        identifier: Identifier to be added to file_name.

    Returns:
        (str): Path with `file_name` plus added identifier.

    """

    file_name = Path(file_name)
    file_name_identifier = file_name.parent.joinpath(file_name.stem + identifier).with_suffix(file_name.suffix)

    return file_name_identifier.as_posix()
