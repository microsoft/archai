# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Utilities that calculates the Pareto-frontier.
"""

from typing import List, Optional

import numpy as np


def find_pareto_frontier_points(all_points: np.ndarray,
                                is_decreasing: Optional[bool] = True) -> List[int]:
    """Takes in a list of n-dimensional points, one per row, returns the list of row indices
        which are Pareto-frontier points.
        
    Assumes that lower values on every dimension are better.

    Args:
        all_points: N-dimensional points.
        is_decreasing: Whether Pareto-frontier decreases or not.

    Returns:
        List of Pareto-frontier indexes.
    
    """

    # For each point see if there exists  any other point which dominates it on all dimensions
    # If that is true, then it is not a pareto point and vice-versa

    # Inputs should alwyas be a two-dimensional array
    assert len(all_points.shape) == 2

    pareto_inds = []

    dim = all_points.shape[1]

    for i in range(all_points.shape[0]):
        this_point = all_points[i,:]
        is_pareto = True

        for j in range(all_points.shape[0]):
            if j == i:
                continue

            other_point = all_points[j,:]

            if is_decreasing:
                diff = this_point - other_point
            else:
                diff = other_point - this_point

            if sum(diff > 0) == dim:
                # Other point is smaller/larger on all dimensions
                # so we have found at least one dominating point
                is_pareto = False
                break
                
        if is_pareto:
            pareto_inds.append(i)

    return pareto_inds
