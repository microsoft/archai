# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


def calc_pareto_frontier(points):
    """ Given an array of points where the first 2 coordinates define a 2D point
    return a list of array indexes that define the pareto frontier for these points """
    pareto = []
    pareto += [0]
    p1 = points[0]
    for i in range(1, len(points)):
        p2 = points[i]
        if p2[1] > p1[1]:
            pareto += [i]
            p1 = p2

    return pareto
