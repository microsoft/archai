# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod

from overrides import EnforceOverrides


class ExperimentRunner(EnforceOverrides):
    """Abstract class for experiment runners.

    The `ExperimentRunner` class provides an abstract interface for creating experiment runners.

    Note:
        This class is inherited from `EnforceOverrides` and any overridden methods in the
        subclass should be decorated with `@overrides` to ensure they are properly overridden.

    """

    def __init__(self) -> None:
        """Initialize the experiment runner."""

        pass

    @abstractmethod
    def run(self) -> None:
        """Run an experiment."""

        pass
