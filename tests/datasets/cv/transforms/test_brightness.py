# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from archai.datasets.cv.transforms.brightness import Brightness, RandomBrightness


def test_brightness():
    # Assert that it works with minimum value
    b = Brightness(-1.0)
    img = torch.tensor([0.5])
    result = b(img)
    assert result.tolist() == [0.0]

    # Assert that it works with maximum value
    b = Brightness(1.0)
    img = torch.tensor([0.5])
    result = b(img)
    assert result.tolist() == [1.0]

    # Assert that it works with value within bounds
    b = Brightness(0.5)
    img = torch.tensor([0.5])
    result = b(img)
    assert result.tolist() == [1.0]


def test_random_brightness():
    # Assert that it works with minimum and maximum values
    rb = RandomBrightness(-1.0, 1.0)
    img = torch.tensor([0.5])
    result = rb(img)
    assert result.tolist()[0] >= -1.0 and result.tolist()[0] <= 1.0
