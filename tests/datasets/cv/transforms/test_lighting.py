# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from archai.datasets.cv.transforms.lighting import Lighting


def test_lighting_transform():
    lighting_transform = Lighting(
        std=0.1,
        eigval=[0.2126, 0.7152, 0.0722],
        eigvec=[[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]],
    )

    # Asser that it produces a different tensor
    img = torch.zeros((3, 256, 256), dtype=torch.float32)
    t_img = lighting_transform(img)
    assert not torch.allclose(img, t_img, rtol=1e-3, atol=1e-3)
