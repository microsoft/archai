# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import gc
import timeit
from typing import Optional


class MeasureBlockTime:
    """"""

    def __init__(self, name: str, disable_gc: Optional[bool] = False, verbose: Optional[bool] = False) -> None:
        """"""

        self.name = name
        self.disable_gc = disable_gc
        self.verbose = verbose

    def __enter__(self):
        """"""

        self.is_gc_enabled = gc.isenabled()

        if self.disable_gc:
            gc.disable()

        self.start_time = timeit.default_timer()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """"""

        if self.disable_gc and self.is_gc_enabled:
            gc.enable()

        if self.verbose:
            print(f"{self.name}: {self.elapsed:.4g} secs")

        return False

    @property
    def elapsed(self) -> float:
        """"""

        return timeit.default_timer() - self.start_time
