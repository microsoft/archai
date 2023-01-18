# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

import torch


class Lighting:
    """Lighting transform."""

    def __init__(self, std: float, eigval: List[float], eigvec: List[float]) -> None:
        """Initializes the lighting transform.

        Args:
            std: Standard deviation of the normal distribution.
            eigval: Eigenvalues of the covariance matrix.
            eigvec: Eigenvectors of the covariance matrix.

        """

        self.std = std
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.std == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.std)
        rgb = (
            self.eigvec.type_as(img)
            .clone()
            .mul(alpha.view(1, 3).expand(3, 3))
            .mul(self.eigval.view(1, 3).expand(3, 3))
            .sum(1)
            .squeeze()
        )

        return img.add(rgb.view(3, 1, 1).expand_as(img))
