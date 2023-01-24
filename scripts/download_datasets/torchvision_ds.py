# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from torchvision import transforms
import torchvision

import dataset_utils

if __name__ == '__main__':
    dataroot = dataset_utils.get_dataroot()
    torchvision.datasets.STL10(root=dataroot,
                                split="train",
                                #train=True,
                                download=True,
                                transform=transforms.Compose([]))
    torchvision.datasets.STL10(root=dataroot,
                                split="test",
                               #train=False,
                                download=True,
                                transform=transforms.Compose([]))
    print('done')