# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from decimal import Decimal
from typing import Dict, Type
import os
import sys
import pathlib

import psutil

from archai.common.config import Config
from archai.common import utils

def _create_ram_disk(req_ram:int, path:str)->bool:
    os.makedirs(path, exist_ok=True)
    return True

    # tmp_filepath = os.path.join(path,'delete_me.temp')
    # disk_speed_command = f'dd if=/dev/zero of="{tmp_filepath}" bs=4k count=100000; rm "{tmp_filepath}"'
    # utils.exec_shell_command(disk_speed_command)

    # avail_mem = psutil.virtual_memory().available
    # print(f'RAM Disk params: req_ram={req_ram}, avail_mem={avail_mem}, path={path}')
    # if avail_mem > req_ram:
    #     utils.exec_shell_command(f'sudo mount -t tmpfs -o size={req_ram} pt_data "{path}"')
    #     utils.exec_shell_command(f'sudo mount') # display mounts
    #     utils.exec_shell_command(disk_speed_command)
    #     return True
    # else:
    #     print('RAM disk is not created because not enough memory')
    #     return False

def untar_dataset(conf_name:str, pt_data_dir:str, conf_dataset:Config, dataroot:str)->None:
    if 'storage_name' not in conf_dataset or not conf_dataset['storage_name']:
        print(f'data config {conf_name} ignored because storage_name key was not found or not set')
        return

    print(f'Untaring for data config: {conf_name}')

    storage_name = conf_dataset['storage_name']
    tar_filepath = os.path.join(pt_data_dir, storage_name + '.tar')
    if not os.path.isfile(tar_filepath):
        raise RuntimeError(f'Tar file for dataset at {tar_filepath} was not found')

    tar_size = pathlib.Path(tar_filepath).stat().st_size
    print('tar_filepath:', tar_filepath, 'tar_size:', tar_size)

    local_dataroot = utils.full_path(dataroot)
    print('local_dataroot:', local_dataroot)
    _create_ram_disk(tar_size, local_dataroot)
    # os.makedirs(local_dataroot, exist_ok=True)

    utils.exec_shell_command(f'tar -xf "{tar_filepath}" -C "{local_dataroot}"')

    print(f'dataset copied from {tar_filepath} to {local_dataroot} sucessfully')


def main():
    parser = argparse.ArgumentParser(description='Archai data install')
    parser.add_argument('--dataroot', type=str, default='~/dataroot',
                        help='path to dataroot on local drive')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Name of the dataset for which confs/dataset/name.yaml should exist and have name of folder or tar file it resides in')
    args, extra_args = parser.parse_known_args()

    pt_data_dir = os.environ.get('PT_DATA_DIR', '')
    if not pt_data_dir:
        raise RuntimeError('This script needs PT_DATA_DIR environment variable with path to dataroot on cloud drive')
    pt_data_dir = utils.full_path(pt_data_dir)
    print('pt_data_dir:', pt_data_dir)

    conf_data_filepath = f'confs/datasets/{args.dataset}.yaml'
    print('conf_data_filepath:', conf_data_filepath)

    conf = Config(config_filepath=conf_data_filepath)
    for dataset_key in ['dataset', 'dataset_search', 'dataset_eval']:
        if dataset_key in conf:
            conf_dataset = conf[dataset_key]
            untar_dataset(dataset_key, pt_data_dir, conf_dataset, args.dataroot)


if __name__ == '__main__':

    # for testing comment below line and set destination path on line 62
    #os.environ['PT_DATA_DIR'] = r'H:\dataroot_cloud'

    main()


