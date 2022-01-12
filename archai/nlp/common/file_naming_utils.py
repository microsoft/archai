# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""File naming utilities to create and modify input identifiers.
"""

from pathlib import Path


def create_file_name_identifier(file_name: Path, identifier: str) -> Path:
    """Adds an identifier (suffix) to the end of the file name.

    Args:
        file_name: Path to have a suffix added.
        identifier: Identifier to be added to file_name.

    Returns:
        (Path): Path with `file_name` plus added identifier.

    """

    return file_name.parent.joinpath(file_name.stem + identifier).with_suffix(file_name.suffix)
