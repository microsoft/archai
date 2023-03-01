import argparse
import os
from archai.datasets.cv.mnist_dataset_provider import MnistDatasetProvider


def main():
    # input and output arguments
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

    for f in os.listdir(path):
        print(f)
        full_path = os.path.join(path, f)
        if os.path.isdir(full_path):
            for g in os.listdir(full_path):
                print(f'  {g}')

    print("end.")
