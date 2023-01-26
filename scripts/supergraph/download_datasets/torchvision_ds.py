# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import dataset_utils
import torchvision
from torchvision import transforms

if __name__ == "__main__":
    dataroot = dataset_utils.get_dataroot()
    torchvision.datasets.STL10(
        root=dataroot,
        split="train",
        # train=True,
        download=True,
        transform=transforms.Compose([]),
    )
    torchvision.datasets.STL10(
        root=dataroot,
        split="test",
        # train=False,
        download=True,
        transform=transforms.Compose([]),
    )
    print("done")
