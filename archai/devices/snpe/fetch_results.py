# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
from test_snpe import download_results
from shutil import rmtree

SNPE_OUTPUT_DIR = 'snpe_output'

files = [x for x in os.listdir('data/test') if x.endswith(".bin")]
files.sort()

output_dir = SNPE_OUTPUT_DIR
if os.path.isdir(output_dir):
    rmtree(output_dir)
os.makedirs(output_dir)

print("Found {} input files".format(len(files)))
download_results(files, 0, output_dir)
