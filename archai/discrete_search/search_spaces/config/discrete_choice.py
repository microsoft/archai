# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, List


class DiscreteChoice:
    """Discrete choice class for search space configuration."""

    def __init__(self, choices: List[Any]) -> None:
        """Initialize the discrete choices.

        Args:
            choices: List of choices.
        """

        self.choices = choices

    def __getitem__(self, idx: str) -> Any:
        return self.choices[idx]

    def __repr__(self) -> str:
        return f"DiscreteChoice({repr(self.choices)})"

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        return len(self.choices)
