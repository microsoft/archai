# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

""" Script to prepare flower102 dataset for pytorch dataloader.
"""

from typing import List, Dict, Tuple, Union, Optional
import os
import pdb
import time
import argparse
import os
import tempfile
import requests

from torchvision.datasets.utils import download_and_extract_archive, download_url
from torch.utils.model_zoo import tqdm

from PIL import Image
import shutil
from collections import defaultdict
import pathlib

from archai.common import utils


def check_flower102(dataroot: str) -> bool:
    flower102 = os.path.join(dataroot, 'flower102')
    train = os.path.join(flower102, 'train')
    test = os.path.join(flower102, 'test')
    meta = os.path.join(flower102, 'meta')

    if not os.path.isdir(flower102) or not os.path.isdir(train) or not os.path.isdir(test) or not os.path.isdir(meta):
        return False

    num_train_files = 0
    for base, dirs, files in os.walk(train):
        for file in files:
            num_train_files += 1

    if num_train_files != 6507:
        return False

    num_test_files = 0
    for base, dirs, files in os.walk(test):
        for file in files:
            num_test_files += 1

    if num_test_files != 1682:
        return False

    # all checks passed
    return True


def download(dataroot: str):
    DOWNLOAD_URL = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
    with tempfile.TemporaryDirectory() as tempdir:
        download_and_extract_archive(
            DOWNLOAD_URL, tempdir, extract_root=dataroot, remove_finished=True)


def load_test_csv_data(filename: str) -> Dict[str, List[str]]:
    ''' Loads the data in csv files into a dictionary with
    class names as keys and list of image names as values. Works only for test data csv'''
    data_dict = defaultdict(list)
    with open(filename, 'r') as f:
        lines = f.readlines()
        assert len(lines) > 0
        for line in lines[1:]:
            words = line.rstrip().split(',')
            assert len(words) > 0
            data_dict[words[0]] = words[1:]

    return data_dict


def load_train_csv_data(filename: str) -> Dict[str, List[str]]:
    ''' Loads the data in csv files into a dictionary with
    class names as keys and list of image names as values. Works only for train data csv '''
    data_dict = defaultdict(list)
    with open(filename, 'r') as f:
        lines = f.readlines()
        assert len(lines) > 0
        for line in lines[1:]:
            words = line.rstrip().split(',')
            assert len(words) > 0
            data_dict[words[1]] = words[2:]

    return data_dict


def copy_data_helper(data: Dict[str, List[str]], imagesroot: str, foldername: str) -> None:

    for key in data.keys():
        images = data[key]
        for im in images:
            if not im:
                continue
            source = os.path.join(imagesroot, im)
            target = os.path.join(foldername, key, im)
            if not os.path.isfile(target):
                utils.copy_file(source, target)


def prepare_data(flower102_root: str):
    test_file = os.path.join(flower102_root, 'meta', 'flowers102_test.csv')
    test_data = load_test_csv_data(test_file)

    # train data is split into 2 files for some reason
    train1_file = os.path.join(flower102_root, 'meta', 'flowers102_train1.csv')
    train2_file = os.path.join(flower102_root, 'meta', 'flowers102_train2.csv')

    train_files = [train1_file, train2_file]
    train_data = defaultdict(list)
    for tf in train_files:
        this_data = load_train_csv_data(tf)
        train_data.update(this_data)

    # make classname directories for train and test
    for key in test_data.keys():
        os.makedirs(os.path.join(flower102_root, 'test', key), exist_ok=True)
        os.makedirs(os.path.join(flower102_root, 'train', key), exist_ok=True)

    # copy images to the right locations
    imagesroot = os.path.join(flower102_root, 'jpg')

    testfoldername = os.path.join(flower102_root, 'test')
    copy_data_helper(test_data, imagesroot, testfoldername)

    trainfoldername = os.path.join(flower102_root, 'train')
    copy_data_helper(train_data, imagesroot, trainfoldername)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='C:\\Users\\dedey\\dataroot',
                        help='root directory where flower102 folder is intended to exist. If it already exists in the format required this script will skip downloading')
    args = parser.parse_args()

    # check that dataset is in format required
    # else download and prepare dataset
    if not check_flower102(args.dataroot):
        # make flower102 directory
        flower102 = os.path.join(args.dataroot, 'flower102')
        train = os.path.join(flower102, 'train')
        test = os.path.join(flower102, 'test')
        meta = os.path.join(flower102, 'meta')

        os.makedirs(flower102, exist_ok=True)
        os.makedirs(train, exist_ok=True)
        os.makedirs(test, exist_ok=True)
        os.makedirs(meta, exist_ok=True)

        # this step will create folder jpg
        # which has all the images
        download(flower102)

        # download the csv files for the train and test split
        # from 'NAS Evaluation is Frustrating' repo
        # note that download_url doesn't work in vscode debug mode
        test_file_url = 'https://raw.githubusercontent.com/antoyang/NAS-Benchmark/master/data/flowers102_test.csv'
        train_file_urls = ['https://raw.githubusercontent.com/antoyang/NAS-Benchmark/master/data/flowers102_train1.csv', 'https://raw.githubusercontent.com/antoyang/NAS-Benchmark/master/data/flowers102_train2.csv']

        download_url(test_file_url, meta, filename=None, md5=None)

        for tu in train_file_urls:
            download_url(tu, meta, filename=None, md5=None)

        prepare_data(flower102)
