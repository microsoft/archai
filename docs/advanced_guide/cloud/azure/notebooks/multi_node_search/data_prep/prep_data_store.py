# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
from archai.datasets.cv.mnist_dataset_provider import MnistDatasetProvider


def main():
    """ This script is in a different folder from the other scripts because this way ensures
    maximum reuse of the output dataset during the development of your other training script.
    Often times those need more debugging and this will save on cloud compute by maximizing
    the reuse of this node in each submitted Azure ML pipeline """
    # input and output arguments
    print("Starting prep_data_store...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="root folder to place the downloaded model")
    args = parser.parse_args()

    path = args.path

    print(f'Writing MNIST dataset to: {path}')

    if not path or not os.path.exists(path):
        raise ValueError(f'Missing path: {path}')

    provider = MnistDatasetProvider(root=path)
    # now force the full download to happen to that root folder.
    provider.get_train_dataset()
    provider.get_val_dataset()
    provider.get_test_dataset()


if __name__ == "__main__":
    main()
