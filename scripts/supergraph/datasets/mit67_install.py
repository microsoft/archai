# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

""" Script to prepare mit67 dataset for pytorch dataloader.
"""

import argparse
import os
import tempfile
from collections import defaultdict
from typing import Dict, List

from torchvision.datasets.utils import download_and_extract_archive, download_url

from archai.supergraph.utils import utils


def check_mit67(dataroot: str) -> bool:
    mit67 = os.path.join(dataroot, "mit67")
    train = os.path.join(mit67, "train")
    test = os.path.join(mit67, "test")
    meta = os.path.join(mit67, "meta")

    if not os.path.isdir(mit67) or not os.path.isdir(train) or not os.path.isdir(test) or not os.path.isdir(meta):
        return False

    num_train_files = 0
    for base, dirs, files in os.walk(train):
        for file in files:
            num_train_files += 1

    if num_train_files != 12466:
        return False

    num_test_files = 0
    for base, dirs, files in os.walk(test):
        for file in files:
            num_test_files += 1

    if num_test_files != 3153:
        return False

    # all checks passed
    return True


def download(dataroot: str):
    DOWNLOAD_URL = "https://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar"
    with tempfile.TemporaryDirectory() as tempdir:
        download_and_extract_archive(DOWNLOAD_URL, tempdir, extract_root=dataroot, remove_finished=True)


def load_test_csv_data(filename: str) -> Dict[str, List[str]]:
    """Loads the data in csv files into a dictionary with
    class names as keys and list of image names as values. Works only for test data csv"""
    data_dict = defaultdict(list)
    with open(filename, "r") as f:
        lines = f.readlines()
        assert len(lines) > 0
        for line in lines[1:]:
            words = line.rstrip().split(",")
            assert len(words) > 0
            data_dict[words[0]] = words[1:]

    return data_dict


def load_train_csv_data(filename: str) -> Dict[str, List[str]]:
    """Loads the data in csv files into a dictionary with
    class names as keys and list of image names as values. Works only for train data csv"""
    data_dict = defaultdict(list)
    with open(filename, "r") as f:
        lines = f.readlines()
        assert len(lines) > 0
        for line in lines[1:]:
            words = line.rstrip().split(",")
            assert len(words) > 0
            data_dict[words[1]] = words[2:]

    return data_dict


def copy_data_helper(data: Dict[str, List[str]], imagesroot: str, foldername: str) -> None:

    for key in data.keys():
        images = data[key]
        for im in images:
            if not im:
                continue
            source = os.path.join(imagesroot, key, im)
            target = os.path.join(foldername, key, im)
            if not os.path.isfile(target):
                utils.copy_file(source, target)


def prepare_data(mit67_root: str):
    test_file = os.path.join(mit67_root, "meta", "MIT67_test.csv")
    test_data = load_test_csv_data(test_file)

    # train data is split into 4 files for some reason
    train1_file = os.path.join(mit67_root, "meta", "MIT67_train1.csv")
    train2_file = os.path.join(mit67_root, "meta", "MIT67_train2.csv")
    train3_file = os.path.join(mit67_root, "meta", "MIT67_train3.csv")
    train4_file = os.path.join(mit67_root, "meta", "MIT67_train4.csv")

    train_files = [train1_file, train2_file, train3_file, train4_file]
    train_data = defaultdict(list)
    for tf in train_files:
        this_data = load_train_csv_data(tf)
        train_data.update(this_data)

    # make classname directories for train and test
    for key in test_data.keys():
        os.makedirs(os.path.join(mit67_root, "test", key), exist_ok=True)
        os.makedirs(os.path.join(mit67_root, "train", key), exist_ok=True)

    # copy images to the right locations
    imagesroot = os.path.join(mit67_root, "Images")

    testfoldername = os.path.join(mit67_root, "test")
    copy_data_helper(test_data, imagesroot, testfoldername)

    trainfoldername = os.path.join(mit67_root, "train")
    copy_data_helper(train_data, imagesroot, trainfoldername)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot",
        type=str,
        default="C:\\Users\\dedey\\dataroot",
        help="root directory where mit67 folder is intended to exist. If mit67 already exists in the format required this script will skip downloading",
    )
    args = parser.parse_args()

    # check that dataset is in format required
    # else download and prepare dataset
    if not check_mit67(args.dataroot):
        # make mit67 directory
        mit67 = os.path.join(args.dataroot, "mit67")
        train = os.path.join(mit67, "train")
        test = os.path.join(mit67, "test")
        meta = os.path.join(mit67, "meta")

        os.makedirs(mit67, exist_ok=True)
        os.makedirs(train, exist_ok=True)
        os.makedirs(test, exist_ok=True)
        os.makedirs(meta, exist_ok=True)

        # this step will create folder mit67/Images
        # which has all the images for each class in its own subfolder
        download(mit67)

        # download the csv files for the train and test split
        # from 'NAS Evaluation is Frustrating' repo
        # note that download_url doesn't work in vscode debug mode
        test_file_url = "https://raw.githubusercontent.com/antoyang/NAS-Benchmark/master/data/MIT67_test.csv"
        train_file_urls = [
            "https://raw.githubusercontent.com/antoyang/NAS-Benchmark/master/data/MIT67_train1.csv",
            "https://raw.githubusercontent.com/antoyang/NAS-Benchmark/master/data/MIT67_train2.csv",
            "https://raw.githubusercontent.com/antoyang/NAS-Benchmark/master/data/MIT67_train3.csv",
            "https://raw.githubusercontent.com/antoyang/NAS-Benchmark/master/data/MIT67_train4.csv",
        ]

        download_url(test_file_url, meta, filename=None, md5=None)

        for tu in train_file_urls:
            download_url(tu, meta, filename=None, md5=None)

        prepare_data(mit67)
