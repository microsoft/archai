# Copyright (c) Microsoft Corporation.
# Licensed under the MIT licen

from dataclasses import dataclass, field
from typing import Any, Dict

import torch


@dataclass
class TorchTrainingArguments:
    """Arguments used in the PyTorch training pipeline.

    Args:
        experiment_name: Name of the experiment.

    """

    experiment_name: str = field(metadata={"help": "Name of the experiment."})

    @property
    def device(self) -> torch.device:
        """Return a PyTorch device instance."""

        return torch.device("cuda" if not self.no_cuda else "cpu")

    def __post_init__(self) -> None:
        """Override post-initialization with custom instructions."""

        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert attributes into a dictionary representation.

        Returns:
            Attributes encoded as a dictionary.

        """

        return self.__dict__
