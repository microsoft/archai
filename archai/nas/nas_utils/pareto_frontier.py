# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Utilities that are related to computing Pareto-frontier.
"""

from typing import List, Optional

import numpy as np


def find_pareto_frontier_points(all_points: np.ndarray,
                                is_decreasing: Optional[bool] = True) -> List[int]:
    """Takes in a list of n-dimensional points, one per row, returns the list of row indices
        which are Pareto-frontier points.
        
    Assumes that lower values on every dimension are better by default.

    Args:
        all_points: N-dimensional points.
        is_decreasing: Whether Pareto-frontier decreases or not.

    Returns:
        List of Pareto-frontier indexes.
    
    """

    # For each point see if there exists  any other point which dominates it on all dimensions
    # If that is true, then it is not a pareto point and vice-versa

    # Inputs should always be a two-dimensional array
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


def compute_crowding_distance(all_points: np.ndarray)->np.ndarray:
    '''Takes in a 2D numpy array, one point per row. Typically these
    are pareto-frontier points or estimate of pareto-frontier points. 
    Assumes each column is an objective in a multi-dimensional optimization problem.
    Computes the crowding distance for each point as detailed in 
    "An effective use of crowding distance in multiobjective particle swarm optimization"
    by Raquel et al., 2005."

    Args:
        all_points: nd.array 2D numpy array.

    Returns: 
        c_dists: nd.array of shape (all_points.shape[0], 1)
    '''

    # Inputs should always be a two-dimensional array
    assert len(all_points.shape) == 2

    assert all_points.shape[0] > 0
    assert all_points.shape[1] > 0

    n = all_points.shape[0]
    num_objs = all_points.shape[1]

    c_dists = np.zeros((all_points.shape[0], 1))

    for i in range(num_objs):
        # sort in ascending order using this objective value
        ids = np.argsort(all_points[:, i])

        # set the distance to be the
        # distance to nearest neighbors on each side
        for j in range(1, n-1):
            row_num = ids[j]
            higher_nbr = ids[j+1]
            lower_nbr = ids[j-1]
            dist_btw_nbrs = all_points[higher_nbr, i] - all_points[lower_nbr, i]
            c_dists[row_num, 1] += dist_btw_nbrs 
            
        # set the maximum distance to the boundary points
        # so that they are always selected
        c_dists[ids[0]] = np.inf
        c_dists[ids[-1]] = np.inf 

     





