# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import cv2
import numpy as np
import glob
import os
import sys
import tqdm
from shutil import rmtree


DEVICE_WORKING_DIR = "/data/local/tmp"


class DataGenerator():
    def __init__(self, root, img_size, subset='quant', count=1000, transpose=None):
        self.img_size = img_size
        self.root = root
        self.subset = subset
        self.transpose = transpose
        all_seg_files = sorted(glob.glob(os.path.join(self.root, '*_seg.png')))
        if len(all_seg_files) == 0:
            print("### no *_seg.png files found in {}".format(self.root))
            sys.exit(1)
        # use first 10000 images for quantization and last 10000 images for test
        assert subset in ['quant', 'test']
        if subset == 'quant':
            self.seg_files = all_seg_files[0:1000]
        elif subset == 'test':
            self.seg_files = all_seg_files[len(all_seg_files) - count:]
        self.img_files = [s.replace("_seg.png", ".png") for s in self.seg_files]

    def __len__(self):
        return len(self.img_files)

    def __call__(self):
        num_imgs = len(self.img_files)
        assert num_imgs > 0
        indices = np.arange(num_imgs)
        for idx in indices:
            img_file = self.img_files[idx]
            img = cv2.imread(img_file)[..., ::-1]     # BGR to RGB
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)
            if self.transpose:
                img = img.transpose(self.transpose)
            yield os.path.basename(img_file), (img / 255).astype(np.float32)


def create_dataset(src_root, dst_root, subset, shape, count, trans=None):
    print(f"Creating {subset} dataset of {count} images with input shape {shape}...")

    image_size = (shape[0], shape[1])
    device_working_dir = DEVICE_WORKING_DIR
    os.makedirs(dst_root, exist_ok=True)

    data_gen = DataGenerator(src_root, image_size, subset, count, trans)

    file_list = []
    with tqdm.tqdm(total=len(data_gen)) as pbar:
        for fname, img in data_gen():
            filename = fname.replace('.png', '.bin')
            path = os.path.join(dst_root, filename)
            file_list.append(filename)
            img.tofile(path)
            pbar.update(1)

    with open(os.path.join(dst_root, 'input_list.txt'), 'w') as f:
        for fname in file_list:
            f.write(fname)
            f.write('\n')

    with open(os.path.join(dst_root, 'input_list_for_device.txt'), 'w') as f:
        for fname in file_list:
            device_path = device_working_dir + '/data/test/' + os.path.basename(fname)
            f.write(device_path)
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create the quant and test datasets')
    parser.add_argument('--input', help='Location of the original input images ' +
                        '(default INPUT_DATASET environment variable')
    parser.add_argument('--count', '-c', type=int, help='Number of images in the test dataset folder ' +
                        '(default 1000)', default=1000)
    parser.add_argument('--dim', '-d', type=int, help='New dimension for the images ' +
                        '(assumes square dimensions, default 256)', default=256)
    parser.add_argument('--transpose', '-t', help="Apply image transpose of '(2, 0 1)'", action="store_true")
    args = parser.parse_args()

    dataset = args.input
    if not dataset:
        dataset = os.getenv("INPUT_DATASET")
        if not dataset:
            print("please provide --input or set your INPUT_DATASET environment variable")
            sys.exit(1)

    count = args.count
    dim = args.dim
    transpose = args.transpose
    if transpose:
        transpose = (2, 0, 1)
    else:
        transpose = None

    dst_root = 'data/quant'
    if os.path.isdir(dst_root):
        rmtree(dst_root)

    create_dataset(dataset, dst_root, 'quant', [dim, dim], count, transpose)
    create_dataset(dataset, dst_root, 'test', [dim, dim], count, transpose)
