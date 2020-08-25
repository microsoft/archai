# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import os
import bisect

def _is_on_ray_left(x1, y1, x2, y2, x3, y3, inclusive=False, epsilon=0):
    """
    Return whether x3,y3 is on the left side of the ray x1,y1 -> x2,y2.
    If inclusive, then the answer is left or on the ray.
    If otherwise, then the answer is strictly left.
    """
    val = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if inclusive:
        return val >= epsilon
    return val > epsilon


def _convex_hull_from_points(xs, ys, eps=None, allow_increase=False):
    """
    Andrew's Monotone Chain Algorithm: (https://en.wikipedia.org/wiki/Graham_scan)

    Assume the data are sorted in order of xs, then the computation complexity is O(n)
    If not sorted, then a sort by x-value is applied first. The complexity becomes O(nlog(n))

    Return:

    hull_indices (list): indices for the points on the hull exactly
    eps_indices (list): indices for the points on the hull + eps tolerance
    """
    indices = list(range(len(xs)))
    if len(xs) <= 1:
        return indices, indices
    # check xs sorted
    is_monotone = True
    for i in range(1, len(xs)):
        if xs[i] < xs[i-1]:
            is_monotone = False
            break
    if not is_monotone:
        indices.sort(key=lambda i : (xs[i], ys[i]))

    def _remove_non_hull_idx(x1, y1, idxs):
        while len(idxs) > 1:
            x2, y2 = xs[idxs[-1]], ys[idxs[-1]]
            x3, y3 = xs[idxs[-2]], ys[idxs[-2]]
            if not _is_on_ray_left(x1, y1, x2, y2, x3, y3):
                if np.abs(x1 - x2) > 1e-6 or np.abs(y1 - y2) > 1e-6:
                    # this ensures that the no points are duplicates
                    break
            del idxs[-1]
        return idxs

    hull_indices = []
    min_y = float('inf')
    for idx in indices:
        x1, y1 = xs[idx], ys[idx]
        min_y = min(y1, min_y)
        hull_indices = _remove_non_hull_idx(x1, y1, hull_indices)
        hull_indices.append(idx)
    if not allow_increase:
        # use a fake final point at (2 * x_max , y_min) to remove increasing.
        x1, y1 = xs[indices[-1]] * 2, min_y
        hull_indices = _remove_non_hull_idx(x1, y1, hull_indices)

    # compute epsilon hull (convex hull + (1+eps) band)
    eps_indices = hull_indices
    if eps is not None and eps > 0:
        eps_indices = []
        h_idx = 0 # right idx, in the hull_indices
        for idx in indices:
            x = xs[idx]
            y = ys[idx]
            if h_idx >= len(hull_indices):
                # Larger than the largest model on the hull
                #y_interp = min_y
                y_interp = ys[hull_indices[-1]]
            elif idx == hull_indices[h_idx]:
                # critical pts on hull
                y_interp = y
                x1, y1 = x, y # hull point to left
                h_idx += 1
                if h_idx < len(hull_indices):
                    x2, y2 = xs[hull_indices[h_idx]], ys[hull_indices[h_idx]]
                else:
                    #x2, y2 = xs[indices[-1]] * 2, min_y
                    x2, y2 = xs[indices[-1]] * 2, ys[hull_indices[-1]]
            else:
                # Between pts of hull
                try:
                    y_interp = y1 + (y2 - y1) / (x2 - x1) * (x - x1)
                    if np.isnan(y_interp):
                        y_interp = min(y1, y2)
                except:
                    # numerical issues when x2, x1 are close
                    y_interp = min(y1, y2)
            if y <= y_interp * (1. + eps):
                eps_indices.append(idx)
                assert x1 <= x and x2 >= x, "idx={} idx[h_idx-1]={} idx[h_idx]={}  x={} y={} x1={} x2={} y1={} y2={} y_interp={}".format(\
                    idx, hull_indices[h_idx-1], hull_indices[h_idx], x, y, x1, x2, y1, y2, y_interp)
    return hull_indices, eps_indices


def _test_convex_hull():
    # random points,
    np.random.seed(19921102)
    xs = np.random.uniform(size=100)
    ys = np.random.uniform(size=100) - xs + 1.0
    eps = np.random.uniform(low=0.0, high=0.3)

    # compute eps convex hull.
    hull_indices, indices = _convex_hull_from_points(xs, ys, eps=eps)

    # plot
    import matplotlib.pyplot as plt
    plt.close('all')

    hull_xs = [xs[i] for i in indices]
    hull_ys = [ys[i] for i in indices]
    bound_xs = [xs[i] for i in hull_indices]
    bound_ys = [ys[i] * (1+eps) for i in hull_indices]
    plt.plot(bound_xs, bound_ys, c='red', label='eps-bound')
    plt.scatter(xs, ys, label='pts')
    plt.scatter(hull_xs, hull_ys, c='black', marker='+', label='eps-hull')
    plt.show()
    #plt.savefig(os.path.join('./temp', 'debug', 'convex_hull.png'),
    #     dpi=plt.gcf().dpi, bbox_inches='tight')


def _convex_hull_insert(hull_xs, hull_ys, x, y, eps=0.0, eg_xs=[], eg_ys=[]):
    """
    Insert a new point (x,y) to a lower convex hull defined by
    hull_xs and hull_ys.

    Assume hull_xs are sorted (increasing).

    returns:
    remove_slice (slice or None) : None if no insert. slice(left, right) if to
    remove the indices left:right
    """
    assert y >= 0, "eps greedy of convex hull need all y > 0"
    # right most point is a fake point (inf, min_y), so the curve is decreasing.
    # left most point is a fake point (0, inf), so that the cheapest is always kept.

    n_xs = len(hull_xs)
    if n_xs == 0:
        # always insert
        return slice(0, 0)

    min_y = np.min(hull_ys)
    idx = bisect.bisect_left(hull_xs, x)
    if idx == n_xs:
        # right most (also covers the n_xs == 1)
        hull_y = min_y

    elif hull_xs[idx] == x:
        hull_y = hull_ys[idx]

    elif idx == 0:
        # left most
        hull_y = y * 1.1
        if hull_y == 0:
            hull_y = 1.0

    else:
        y1, y2 = hull_ys[idx-1:idx+1]
        x1, x2 = hull_xs[idx-1:idx+1]
        # linear interpolate
        hull_y = y1 + (y2 - y1) / (x2 - x1) * (x - x1)

    diff = ( y - hull_y ) / hull_y
    if diff > eps:
        # do not insert
        return None
    if diff >= 0:
        # we insert b/c of eps-greedy. Or there are three points on the same line.
        return slice(n_xs, n_xs)
    # now diff < 0
    # fix left side
    slice_left = idx
    for li in reversed(range(1, idx)):
        x3, x2 = hull_xs[li-1:li+1]
        y3, y2 = hull_ys[li-1:li+1]
        to_rm_li = _is_on_ray_left(x, y, x2, y2, x3, y3, inclusive=False)
        if to_rm_li:
            slice_left = li
        else:
            break
    # fix right side
    slice_right = idx
    min_y = min(y, min_y)
    for li in range(idx, n_xs):
        if li < n_xs - 1:
            x2, x3 = hull_xs[li:li+2]
            y2, y3 = hull_ys[li:li+2]
        else:
            x2, x3 = hull_xs[li], hull_xs[li] * 2
            y2, y3 = hull_ys[li], min_y
        to_rm_li = not _is_on_ray_left(x, y, x2, y2, x3, y3, inclusive=True)
        if to_rm_li:
            slice_right = li + 1
        else:
            break
    # TODO epsilon greedy
    return slice(slice_left, slice_right)


def _test_convex_hull_insert():
    import argparse
    import matplotlib.pyplot as plt
    plt.close('all')
    parser = argparse.ArgumentParser()
    parser.add_argument('--xs', type=str)
    parser.add_argument('--ys', type=str)
    parser.add_argument('-x', type=float)
    parser.add_argument('-y', type=float)
    args = parser.parse_args()

    y_max = 1.0
    y_min = 0.01

    def _random_convex_perf_curve(n):
        slopes = np.random.uniform(low=-3, high=-1e-2, size=n-1)
        slopes.sort()

        delta_ys = np.random.uniform(size=n-1)
        delta_ys = delta_ys / np.sum(delta_ys) * (-y_max + y_min)
        delta_xs = delta_ys / slopes
        xs = [1] + list(np.cumsum(delta_xs) + 1)
        ys = [y_max] + list(np.cumsum(delta_ys) + y_max)
        return xs, ys

    if args.xs and args.ys:
        xs = list(map(float, args.xs.split(',')))
        ys = list(map(float, args.ys.split(',')))
    else:
        xs, ys = _random_convex_perf_curve(8)
    print(xs)
    print(ys)
    plt.plot(xs + [xs[-1] * 2.0], ys + [y_min], color='r', marker='x')
    n = len(xs)

    scater_xs, scater_ys = [], []
    if args.x and args.y:
        x = args.x
        y = args.y
        scater_xs.append(x)
        scater_ys.append(y)

    locs = [0, 1, n//2, n-2, n-1]
    for i in locs:
        if i < n-1:
            x = (xs[i] + xs[i+1]) / 2.
            y = max(y_min / 2, min(y_max, (1 + np.random.uniform(-0.9, -0.3)) * (ys[i] + ys[i+1]) / 2.))
        else:
            x = xs[i] * 1.5
            y = max(y_min / 2, min(y_max, (1 + np.random.uniform(-0.9, -0.3)) * ys[i]))
        scater_xs.append(x)
        scater_ys.append(y)

    x = (2 * xs[-1] + xs[-2]) / 3
    y = y_min / 2
    scater_xs.append(x)
    scater_ys.append(y)
    for x, y in zip(scater_xs, scater_ys):
        ret = _convex_hull_insert(xs, ys, x, y)
        print("x={} y={} ret={}".format(x, y, ret))

    plt.scatter(scater_xs, scater_ys)
    plt.savefig(os.path.join('./temp', 'debug', 'convex_hull_insert.png'),
        dpi=plt.gcf().dpi, bbox_inches='tight')


if __name__ == '__main__':
    _test_convex_hull()