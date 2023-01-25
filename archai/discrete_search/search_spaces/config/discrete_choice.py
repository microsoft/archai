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
        """Get the choice at the given index.

        Args:
            idx: Index of the choice.

        Returns:
            Choice at the given index.

        """

        return self.choices[idx]

    def __repr__(self) -> str:
        """Get the print representation of the discrete choice.

        Returns:
            Print representation of the discrete choice.

        """

        return f"DiscreteChoice({repr(self.choices)})"

    def __str__(self) -> str:
        """Get the string representation of the discrete choice.

        Returns:
            String representation of the discrete choice.

        """

        return self.__repr__()

    def __len__(self) -> int:
        """Get the number of choices.

        Returns:
            Number of choices.

        """

        return len(self.choices)
