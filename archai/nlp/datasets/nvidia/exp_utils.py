# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for creating experiment-related paths.
"""

import os
import shutil

from typing import Optional, Tuple


from archai.common import utils, common


def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
    os.makedirs(dir_path, exist_ok=True)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        os.makedirs(script_path, exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def dataset_dir_name(dataset:str)->str:
    if dataset=='wt103':
        return 'wikitext-103'
    if dataset=='wt2':
        return 'wikitext-2'
    if dataset.startswith('olx_'):
        return dataset
    if dataset=='lm1b':
        return 'one-billion-words'
    if dataset=='enwik8':
        raise RuntimeError(f'dataset "{dataset}" is not supported yet')
    if dataset=='text8':
        raise RuntimeError(f'dataset "{dataset}" is not supported yet')
    raise RuntimeError(f'dataset "{dataset}" is not known')


def get_create_dirs(dataroot:Optional[str], dataset_name:str,
                    experiment_name='nv_xformer_xl', output_dir='~/logdir',
                    pretrained_path:Optional[str]="", cache_dir:Optional[str]="")->Tuple[str,str,str,str,str]:

    pt_data_dir, pt_output_dir = common.pt_dirs()
    if pt_output_dir:
        pt_output_dir = os.path.join(pt_output_dir, experiment_name)
    dataroot = dataroot or pt_data_dir or common.default_dataroot()
    dataroot = utils.full_path(dataroot)

    dataset_dir = utils.full_path(os.path.join(dataroot,'textpred', dataset_dir_name(dataset_name)))
    output_dir = utils.full_path(pt_output_dir or \
                        os.path.join(output_dir, experiment_name)
                    , create=True)

    if not os.path.isabs(cache_dir):
        cache_dir = os.path.join(dataset_dir, cache_dir)
    
    cache_dir = utils.full_path(cache_dir, create=True)

    if not os.path.isabs(pretrained_path) and pretrained_path:
        pretrained_path = os.path.join(os.path.dirname(output_dir), pretrained_path)

    return dataset_dir, output_dir, pretrained_path, cache_dir, dataroot
