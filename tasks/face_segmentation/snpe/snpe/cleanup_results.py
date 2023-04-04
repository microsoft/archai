# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import sys
import glob
from shutil import copyfile, rmtree


def cleanup_results(dataset, output_dir):
    """ cleans up a folder containing Results* folders downloaded from device
    and renames the output images to match the input image names so that
    collect_metrics can work on the folder """

    test_size = 0
    result_dirs = os.listdir(output_dir)
    for name in result_dirs:
        if name.startswith('Result_'):
            test_size += 1

    print(f"Found {test_size} Result_* folders")

    all_seg_files = sorted(glob.glob(os.path.join(dataset, '*_seg.png')))
    if len(all_seg_files) < test_size:
        print(f"### not enough *_seg.png files found in {dataset}")
        sys.exit(1)

    test_files = all_seg_files[len(all_seg_files) - test_size:]

    raw_file_name = 'output.raw'
    index = 0
    for name in test_files:
        name = os.path.basename(name).split('_')[0]
        result = f"Result_{index}"
        if result in result_dirs:
            raw_file = os.path.join(output_dir, result, raw_file_name)
            if not os.path.isfile(raw_file):
                raw_file_name = os.listdir(os.path.join(output_dir, result))[0]
                raw_file = os.path.join(output_dir, result, raw_file_name)
            output_file = os.path.join(output_dir, name + '.raw')
            print(f"{raw_file} ===> {output_file}")
            copyfile(raw_file, output_file)
            rmtree(os.path.join(output_dir, result))
        index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check the outputs from the device')
    parser.add_argument('--input', help='Location of the original input images ' +
                        '(defaults to INPUT_DATASET environment variable)')
    parser.add_argument('--output', '-o', help='Location of the folder containing Result* folders')
    args = parser.parse_args()
    dataset = args.input
    if not dataset:
        dataset = os.getenv("INPUT_DATASET")
        if not dataset:
            print("please provide --input or set your INPUT_DATASET environment vairable")
            sys.exit(1)

    output = args.output
    if not os.path.isdir(output):
        print("--output dir not found")
        sys.exit(1)

    cleanup_results(dataset, output)
    print("Done")
