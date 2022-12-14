import numpy as np
from scipy import interpolate
from scipy import spatial
from functools import reduce


def spline_fit(xs, ys):
    indexes = np.argsort(np.array(xs), axis=0)
    ixs = []
    iys = []
    for i in indexes:
        ixs += [xs[i]]
        iys += [ys[i]]

    f = interpolate.UnivariateSpline(ixs, iys)
    f.set_smoothing_factor(0.1)
    xn = np.linspace(ixs[0], ixs[-1], 100)
    return [xn, f(xn)]


def filter_(pts, pt):
    """
    Get all points in pts that are not Pareto dominated by the point pt
    """
    weakly_worse = (pts <= pt).all(axis=-1)
    strictly_worse = (pts < pt).any(axis=-1)
    return pts[~(weakly_worse & strictly_worse)]


def get_pareto_undominated_by(pts1, pts2=None):
    """
    Return all points in pts1 that are not Pareto dominated
    by any points in pts2
    """
    if pts2 is None:
        pts2 = pts1
    return reduce(filter_, pts2, pts1)


def get_pareto_frontier(pts):
    """
    Iteratively filter points based on the convex hull heuristic
    """
    pareto_groups = []

    # loop while there are points remaining
    while pts.shape[0]:
        # brute force if there are few points:
        if pts.shape[0] < 10:
            pareto_groups.append(get_pareto_undominated_by(pts))
            break

        # compute vertices of the convex hull
        hull_vertices = spatial.ConvexHull(pts).vertices

        # get corresponding points
        hull_pts = pts[hull_vertices]

        # get points in pts that are not convex hull vertices
        nonhull_mask = np.ones(pts.shape[0], dtype=bool)
        nonhull_mask[hull_vertices] = False
        pts = pts[nonhull_mask]

        # get points in the convex hull that are on the Pareto frontier
        pareto = get_pareto_undominated_by(hull_pts)
        pareto_groups.append(pareto)

        # filter remaining points to keep those not dominated by
        # Pareto points of the convex hull
        pts = get_pareto_undominated_by(pts, pareto)

    return np.vstack(pareto_groups)


def pareto_curve(Xs, Ys):
    # in our case faster is better so invert the X axis
    pts = np.array(list(zip(Xs, Ys))) * np.array((-1, 1))
    result = get_pareto_frontier(pts)
    indices = []
    for pt in result:
        i = np.where(pts == pt)[0][0]
        indices += [i]
    return indices


def get_pareto_edges(xs, ys):
    left = None
    top = None
    bottom = None
    right = None
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        if left is None or x < left:
            left = x
        if bottom is None or y < bottom:
            bottom = y
        if top is None or y > top:
            top = y
        if right is None or x > right:
            right = x

    return (left, bottom, right, top)
