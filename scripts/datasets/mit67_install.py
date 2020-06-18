# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

""" Script to prepare food101 dataset for pytorch dataloader.

This script assumes that one has downloaded and extracted the full food101 dataset from ETHZ.
Invoke the script as $ python food101.py --rootdir /path/to/food101. It will create 'train' and 'test'
folders inside the root folder filled with the official train and test splits. The folder
structure in 'train' and 'test' respect that needed for pytorch torchvision.datasets.ImageFolder
to work.

"""

import os
import pdb
import time
import argparse
import os
import tempfile

from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.model_zoo import tqdm

from PIL import Image
import shutil
from collections import defaultdict
import pathlib


def download(dataroot:str):
    DOWNLOAD_URL = 'http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar'
    download_and_extract_archive(DOWNLOAD_URL, tempfile.tempdir, extract_root=dataroot, remove_finished=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='d:\\datasets',
                        help='root directory where mit67 folder is intended to exist. If mit67 already exists in the format required this script will skip downloading')

    # check that dataset is in format required
    # else download and prepare dataset
    if not check_mit67(args.dataroot):


