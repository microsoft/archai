# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# https://github.com/quark0/darts/blob/master/cnn/utils.py

import numpy as np
import torch


class CustomCutout:
    """Custom-based cutout transform."""

    def __init__(self, length: int) -> None:
        """Initialize the custom-based cutout transform.

        Args:
            length: Length of the cutout.

        """

        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)

        img *= mask

        return img
