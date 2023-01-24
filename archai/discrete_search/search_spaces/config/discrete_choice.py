# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, List


class DiscreteChoice:
    def __init__(self, choices: List[Any]):
        self.choices = choices

    def __getitem__(self, idx):
        return self.choices[idx]

    def __repr__(self):
        return f"DiscreteChoice({repr(self.choices)})"

    def __str__(self):
        return self.__repr__()

    def __len__(self) -> int:
        return len(self.choices)
