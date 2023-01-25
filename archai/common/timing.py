# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import gc
import timeit
from types import TracebackType
from typing import Optional


class MeasureBlockTime:
    """Context manager that measures the time elapsed in a block of code."""

    def __init__(self, name: str, disable_gc: Optional[bool] = False, verbose: Optional[bool] = False) -> None:
        """Initilize the timer.

        Args:
            name: Name of the timer.
            disable_gc: Whether to disable the garbage collector during the time measurement.
            verbose: Whether to print the elapsed time when exiting the context manager.

        """

        self.name = name
        self.disable_gc = disable_gc
        self.verbose = verbose

    def __enter__(self) -> MeasureBlockTime:
        self.is_gc_enabled = gc.isenabled()

        if self.disable_gc:
            gc.disable()

        self.start_time = timeit.default_timer()

        return self

    def __exit__(self, exc_type: type[BaseException], exc_val: BaseException, exc_tb: TracebackType) -> None:
        if self.disable_gc and self.is_gc_enabled:
            gc.enable()

        if self.verbose:
            print(f"{self.name}: {self.elapsed:.4g} secs")

        return False

    @property
    def elapsed(self) -> float:
        """Return the elapsed time in seconds."""

        return timeit.default_timer() - self.start_time
