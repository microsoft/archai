# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from archai.datasets.cv.transforms.custom_cutout import CustomCutout


def test_custom_cutout():
    # Assert that it works with length argument
    c = CustomCutout(length=10)
    img = torch.ones((3, 20, 20))

    result = c(img)
    non_zero_elements = (result == 0).sum().item()
    assert non_zero_elements > 0

    # Assert that it produces the different result (due to randomness) for the same image
    c1 = CustomCutout(length=10)
    c2 = CustomCutout(length=10)

    img1 = torch.ones((3, 20, 20))
    img2 = torch.ones((3, 20, 20))

    result1 = c1(img1)
    result2 = c2(img2)
    assert not torch.equal(result1, result2)
