# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import gc
import time

from archai.common.timing import MeasureBlockTime


def test_measure_block_time():
    # Assert the basic functionality of the timer
    with MeasureBlockTime("test_timer") as timer:
        time.sleep(0.5)
        elapsed_time = timer.elapsed
    assert elapsed_time >= 0.45

    # Assert if the garbage collector is disabled during the time measurement
    with MeasureBlockTime("test_timer", disable_gc=True) as timer:
        time.sleep(0.5)
        elapsed_time = timer.elapsed
        assert gc.isenabled() is False
    assert elapsed_time >= 0.45

    # Assert if the garbage collector is enabled after the time measurement
    with MeasureBlockTime("test_timer", disable_gc=True) as timer:
        time.sleep(0.5)
        elapsed_time = timer.elapsed
    assert gc.isenabled() is True
