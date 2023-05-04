# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy as np


def calc_pareto_frontier(points):
    """ Given an array of points where the first 2 coordinates define a 2D point
    return a sorted version of those points and a list of array indexes into that
    sorted list that define the pareto frontier for these points """
    points = np.array(points)
    sorted = points[points[:, 0].argsort()]
    pareto = []
    pareto += [0]
    p1 = sorted[0]
    for i in range(1, len(sorted)):
        p2 = sorted[i]
        if p2[1] > p1[1]:
            pareto += [i]
            p1 = p2

    return (sorted, pareto)
