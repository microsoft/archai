# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from typing import Tuple

import torch


class Brightness:
    """Brightness transform."""

    def __init__(self, value: float) -> None:
        """Initializes the brightness transform.

        Args:
            value: Brightness factor, e.g., 0 = no change, 1 = completely white,
                -1 = completely black, <0 = darker, >0 = brighter.

        """

        self.value = max(min(value, 1.0), -1.0)

    def __call__(self, *imgs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        outputs = []

        for idx, img in enumerate(imgs):
            img = torch.clamp(img.float().add(self.value).type(img.type()), 0, 1)
            outputs.append(img)

        return outputs if idx > 1 else outputs[0]


class RandomBrightness:
    """Random brightness transform."""

    def __init__(self, min_val: float, max_val: float) -> None:
        """ "Initializes the random brightness transform.

        Args:
            min_val: Minimum brightness factor.
            max_val: Maximum brightness factor.

        """

        self.values = (min_val, max_val)

    def __call__(self, *imgs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        value = random.uniform(self.values[0], self.values[1])
        outputs = Brightness(value)(*imgs)

        return outputs
