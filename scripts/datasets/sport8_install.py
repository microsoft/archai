# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

""" Script to prepare sport8 dataset for pytorch dataloader. On Windows requires installation of WinRAR
from here: https://www.rarlab.com/download.htm and adding path of unrar.exe to PATH environment variable.
"""

from scripts.datasets.mit67_install import load_train_csv_data
from typing import List, Dict, Tuple, Union, Optional
import os
import pdb
import time
import argparse
import os
import tempfile
import requests
import rarfile

from torchvision.datasets.utils import download_and_extract_archive, download_url
from torch.utils.model_zoo import tqdm

from PIL import Image
import shutil
from collections import defaultdict
import pathlib


def load_csv_data(filename: str) -> Dict[str, List[str]]:
    ''' Loads the data in csv files into a dictionary with 
    class names as keys and list of image names as values. '''
    data_dict = defaultdict(list)
    with open(filename, 'r') as f:
        lines = f.readlines()
        assert len(lines) > 0
        for line in lines[1:]:
            words = line.rstrip().split(',')
            assert len(words) > 0
            data_dict[words[0]] = words[1:]

    return data_dict


def check_sport8(dataroot: str) -> bool:
    mit67 = os.path.join(dataroot, 'sport8')
    train = os.path.join(mit67, 'train')
    test = os.path.join(mit67, 'test')
    meta = os.path.join(mit67, 'meta')

    if not os.path.isdir(sport8) or not os.path.isdir(train) or not os.path.isdir(test) or not os.path.isdir(meta):
        return False

    num_train_files = 0
    for base, dirs, files in os.walk(train):
        for file in files:
            num_train_files += 1

    if num_train_files != 1261:
        return False

    num_test_files = 0
    for base, dirs, files in os.walk(test):
        for file in files:
            num_test_files += 1

    if num_test_files != 318:
        return False

    # all checks passed
    return True


def download(dataroot:str):
    DOWNLOAD_URL = 'http://vision.stanford.edu/lijiali/event_dataset/event_dataset.rar'
    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.basename(DOWNLOAD_URL)
        download_url(DOWNLOAD_URL, tempdir, filename)
        archive = os.path.join(tempdir, filename)
        print(f"Extracting {archive} to {dataroot}")
        r = rarfile.RarFile(archive, mode="r")
        r.extractall(path=dataroot)


def copy_data_helper(data: Dict[str, List[str]], imagesroot: str, foldername: str) -> None:

    for key in data.keys():
        images = data[key]
        for im in images:
            if not im:
                continue
            source = os.path.join(imagesroot, key, im)
            target = os.path.join(foldername, key, im)
            if not os.path.isfile(target):
                shutil.copy2(source, target)


def prepare_data(sport8_root: str):
    test_file = os.path.join(sport8_root, 'meta', 'Sport8_test.csv')
    test_data = load_csv_data(test_file)

    train_file = os.path.join(sport8_root, 'meta', 'Sport8_train.csv')
    train_data = load_csv_data(train_file)

    # make classname directories for train and test
    for key in test_data.keys():
        os.makedirs(os.path.join(sport8_root, 'test', key), exist_ok=True)
        os.makedirs(os.path.join(sport8_root, 'train', key), exist_ok=True)

    # copy images to the right locations
    imagesroot = os.path.join(sport8_root, 'event_img')

    testfoldername = os.path.join(sport8_root, 'test')
    copy_data_helper(test_data, imagesroot, testfoldername)

    trainfoldername = os.path.join(sport8_root, 'train')
    copy_data_helper(train_data, imagesroot, trainfoldername)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='C:\\Users\\dedey\\dataroot',
                        help='root directory where mit67 folder is intended to exist. If sport8 already exists in the format required this script will skip downloading')
    args = parser.parse_args()

    # check that dataset is in format required
    # else download and prepare dataset
    if not check_sport8(args.dataroot):
        # make sport8 directory
        sport8 = os.path.join(args.dataroot, 'sport8')
        train = os.path.join(sport8, 'train')
        test = os.path.join(sport8, 'test')
        meta = os.path.join(sport8, 'meta')

        os.makedirs(sport8, exist_ok=True)
        os.makedirs(train, exist_ok=True)
        os.makedirs(test, exist_ok=True)
        os.makedirs(meta, exist_ok=True)

        # this step will create folder sport8/event_img
        # which has all the images for each class in its own subfolder
        download(sport8)

        # download the csv files for the train and test split
        # from 'NAS Evaluation is Frustrating' repo
        # note that download_url doesn't work in vscode debug mode
        test_file_url = 'https://raw.githubusercontent.com/antoyang/NAS-Benchmark/master/data/Sport8_test.csv'
        train_file_url = 'https://raw.githubusercontent.com/antoyang/NAS-Benchmark/master/data/Sport8_train.csv'

        download_url(test_file_url, meta, filename=None, md5=None)
        download_url(train_file_url, meta, filename=None, md5=None)
        
        prepare_data(sport8)
