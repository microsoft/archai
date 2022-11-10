# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

""".
"""

import os
from typing import Optional, Tuple


from archai.common import utils, common


def get_dataset_dir(dataset: str) -> str:
    """
    """

    if dataset == "wt2":
        return "wikitext-2"
    if dataset == "wt103":
        return "wikitext-103"
    if dataset=="lm1b":
        return "one-billion-words"
    if dataset.startswith("olx_"):
        return dataset

    raise RuntimeError(f"Dataset: {dataset} is not supported yet.")


def get_create_dirs(dataroot:Optional[str], dataset_name:str,
                    experiment_name='nv_xformer_xl', output_dir='~/logdir',
                    pretrained_path:Optional[str]="", cache_dir:Optional[str]="")->Tuple[str,str,str,str,str]:

    pt_data_dir, pt_output_dir = common.pt_dirs()
    if pt_output_dir:
        pt_output_dir = os.path.join(pt_output_dir, experiment_name)
    dataroot = dataroot or pt_data_dir or common.default_dataroot()
    dataroot = utils.full_path(dataroot)

    dataset_dir = utils.full_path(os.path.join(dataroot,'textpred', get_dataset_dir(dataset_name)))
    output_dir = utils.full_path(pt_output_dir or \
                        os.path.join(output_dir, experiment_name)
                    , create=True)

    if not os.path.isabs(cache_dir):
        cache_dir = os.path.join(dataset_dir, cache_dir)
    
    cache_dir = utils.full_path(cache_dir, create=True)

    if not os.path.isabs(pretrained_path) and pretrained_path:
        pretrained_path = os.path.join(os.path.dirname(output_dir), pretrained_path)

    return dataset_dir, output_dir, pretrained_path, cache_dir, dataroot

