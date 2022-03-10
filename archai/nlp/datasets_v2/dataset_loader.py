# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Auxiliary functions that allows loading datasets from files or hub.
"""

from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset
from datasets.utils.download_manager import GenerateMode


def _should_refresh_cache(refresh: bool) -> GenerateMode:
    """
    """

    if refresh:
        return GenerateMode.FORCE_REDOWNLOAD

    return GenerateMode.REUSE_DATASET_IF_EXISTS


def load_file_dataset(data_dir: str,
                      cache_dir: str,
                      data_files: Optional[Union[Dict[str, Any], List[str]]] = None,
                      split: Optional[List[str]] = None,
                      features: Optional[List[str]] = None,
                      from_stream: Optional[bool] = False,
                      refresh_cache: Optional[bool] = False):
    """
    """

    return load_dataset(data_dir,
                        cache_dir=cache_dir,
                        data_files=data_files,
                        split=split,
                        download_mode=_should_refresh_cache(refresh_cache),
                        features=features,
                        streaming=from_stream)


def load_hub_dataset(data_dir: str,
                     data_config_name: str,
                     cache_dir: str,
                     split: Optional[List[str]] = None,
                     revision: Optional[List[str]] = None,
                     from_stream: Optional[bool] = False,
                     refresh_cache: Optional[bool] = False):
    """
    """

    return load_dataset(data_dir,
                        name=data_config_name,
                        cache_dir=cache_dir,
                        split=split,
                        download_mode=_should_refresh_cache(refresh_cache),
                        revision=revision,
                        streaming=from_stream)
