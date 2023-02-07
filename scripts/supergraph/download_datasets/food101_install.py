# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

""" Script to prepare food101 dataset for pytorch dataloader.

This script assumes that one has downloaded and extracted the full food101 dataset from ETHZ.
Invoke the script as $ python food101.py --rootdir /path/to/food101. It will create 'train' and 'test'
folders inside the root folder filled with the official train and test splits. The folder
structure in 'train' and 'test' respect that needed for pytorch torchvision.datasets.ImageFolder
to work.

"""

import argparse
import os
import pathlib
import tempfile

from torch.utils.model_zoo import tqdm
from torchvision.datasets.utils import download_and_extract_archive

from archai.common import utils


def copy_file_list(file_list, src_dir, dest_dir):
    with tqdm(total=len(file_list)) as pbar:
        for i, filename in enumerate(file_list):
            filename = filename.strip()
            if filename:
                # convert / to os-specific dir separator
                filename_parts = (filename + ".jpg").split("/")
                target = os.path.join(dest_dir, *filename_parts)
                if not os.path.isfile(target):
                    utils.copy_file(os.path.join(src_dir, *filename_parts), target)
            pbar.update(1)


def prepare_data(dataroot: str) -> None:
    meta_path = os.path.join(dataroot, "food-101", "meta")
    images_path = os.path.join(dataroot, "food-101", "images")
    train_path = os.path.join(dataroot, "food-101", "train")
    test_path = os.path.join(dataroot, "food-101", "test")

    train_list = pathlib.Path(os.path.join(meta_path, "train.txt")).read_text().splitlines()
    test_list = pathlib.Path(os.path.join(meta_path, "test.txt")).read_text().splitlines()
    class_list = pathlib.Path(os.path.join(meta_path, "classes.txt")).read_text().splitlines()

    for class_name in class_list:
        class_name = class_name.strip()
        if class_name:
            os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_path, class_name), exist_ok=True)

    copy_file_list(train_list, images_path, train_path)
    copy_file_list(test_list, images_path, test_path)


def download(dataroot: str):
    DOWNLOAD_URL = "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    download_and_extract_archive(DOWNLOAD_URL, tempfile.tempdir, extract_root=dataroot, remove_finished=True)


if __name__ == "__main__":
    # download()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot",
        type=str,
        default="d:\\datasets",
        help="root directory where food-101 folder exist (downloaded and extracted from ETHZ)",
    )
    args = parser.parse_args()

    prepare_data(args.dataroot)
