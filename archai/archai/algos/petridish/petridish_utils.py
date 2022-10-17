# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Iterator, Mapping, Type, Optional, Tuple, List
from enum import Enum
import os
import math
import random

import bisect
import numpy as np

import tensorwatch as tw
from tensorwatch import ModelStats
import yaml
import matplotlib.pyplot as plt

from archai.nas.model_desc import ConvMacroParams, CellDesc, CellType, OpDesc, \
                                  EdgeDesc, TensorShape, TensorShapes, NodeDesc, ModelDesc
from archai.common.metrics import Metrics
from archai.common.common import logger, utils

class JobStage(Enum):
    # below values must be assigned in sequence so getting next job stage enum is easy
    SEED = 1
    SEED_TRAINED = 2
    SEARCH = 3
    SEARCH_TRAINED = 4
    EVAL = 5
    EVAL_TRAINED = 6

class ConvexHullPoint:
    def __init__(self, job_stage:JobStage, parent_id:int,
                 sampling_count:int,
                 model_desc:ModelDesc,
                 cells_reductions_nodes:Tuple[int, int, int],
                 metrics:Optional[Metrics]=None,
                 model_stats:Optional[tw.ModelStats]=None) -> None:
        # we only record points after training
        self.job_stage = job_stage
        self.parent_id = parent_id
        self.sampling_count = sampling_count
        self.model_desc = model_desc
        self.cells_reductions_nodes = cells_reductions_nodes
        self.metrics = metrics
        self.model_stats = model_stats

        # TODO: we use random IDs because with ray multiprocessing, its harder to have global
        # id generation. Ideally we should use UUID or global store but for now we just
        # use large enough random range
        ConvexHullPoint._id = random.randint(0, 2147483648)
        self.id = ConvexHullPoint._id

    def is_trained_stage(self)->bool:
        return self.job_stage==JobStage.SEARCH_TRAINED or self.job_stage==JobStage.SEED_TRAINED

    def next_stage(self)->JobStage:
        return JobStage(self.job_stage.value+1)

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

def model_descs_on_front(hull_points:List[ConvexHullPoint], convex_hull_eps:float,
                         lower_hull:bool=True)\
        ->Tuple[List[ConvexHullPoint], List[ConvexHullPoint], List[float], List[float]]:
    assert(len(hull_points) > 0)

    xs = [point.model_stats.MAdd for point in hull_points]
    ys = [1.0-point.metrics.best_val_top1() if lower_hull else point.metrics.best_val_top1()
            for point in hull_points]

    hull_indices, eps_indices = _convex_hull_from_points(xs, ys, eps=convex_hull_eps)
    eps_points = [hull_points[i] for i in eps_indices]
    front_points = [hull_points[i] for i in hull_indices]

    return front_points, eps_points, xs, ys

def hull_points2tsv(points:List[ConvexHullPoint])->str:
    lines = ['\t'.join(['id', 'job_stage',
                        'cells', 'reductions', 'nodes',
                        'MAdd', 'flops', 'duration',
                        'mread', 'mwrite',
                        'inference_memory', 'parameters',
                        'train_best_epoch', 'train_best_top1',
                        'test_best_epoch', 'test_best_top1',
                        'paent_id', 'sampling_count'])]

    for p in points:
        cells, reductions, nodes = p.cells_reductions_nodes
        mstats, metrics = p.model_stats, p.metrics

        vals = []
        vals.extend([p.id, JobStage(p.job_stage).name])

        # add macro
        vals.extend([cells, reductions, nodes])

        # add model stats
        vals.extend([mstats.MAdd, mstats.Flops, mstats.duration])
        vals.extend([mstats.mread, mstats.mwrite])
        vals.extend([mstats.inference_memory, mstats.parameters])

        # add metrics
        train_metrics, val_metrics, test_metrics = metrics.run_metrics.best_epoch()
        vals.extend([train_metrics.index, train_metrics.top1.avg])
        if val_metrics:
            vals.extend([val_metrics.index, val_metrics.top1.avg])
        else:
            vals.extend([math.nan, math.nan])

        # other attributes
        vals.extend([p.parent_id, p.sampling_count])

        line = '\t'.join([str(v) for v in vals])
        lines.append(line)

    return '\n'.join(lines)

def sample_from_hull(hull_points:List[ConvexHullPoint], convex_hull_eps:float)->ConvexHullPoint:
    front_points, eps_points, xs, ys = model_descs_on_front(hull_points,
        convex_hull_eps)

    logger.info(f'num models in pool: {len(hull_points)}')
    logger.info(f'num models on front: {len(front_points)}')
    logger.info(f'num models on front with eps: {len(eps_points)}')

    # form scores to non-maxima supress models already sampled
    counts = [point.sampling_count for point in eps_points]
    counts_max = max(counts)
    counts_min = min(counts)
    if counts_max == counts_min:
        counts_range = counts_max
    else:
        counts_range = counts_max - counts_min
    # to prevent division by 0
    if counts_range == 0:
        counts_range = 1
    # scale between [0,1] to avoid numerical issues
    scaled_counts = [(count - counts_min)/counts_range for count in counts]
    count_scores = [1.0/(scaled_count + 1) for scaled_count in scaled_counts]

    # form scores to sample inversely proportional to madds
    # since it takes less compute to train a smaller model
    # this allows us to evaluate each point equal number of times
    # with any compute budget
    eps_madds = [point.model_stats.MAdd for point in eps_points]
    madd_max = max(eps_madds)
    madd_min = min(eps_madds)
    if madd_max == madd_min:
        madd_range = madd_max
    else:
        madd_range = madd_max - madd_min
    # to prevent division by 0
    if madd_range == 0:
        madd_range = 1
    # scale between [0,1] to avoid numerical issues
    scaled_madds = [(madd - madd_min)/madd_range for madd in eps_madds]
    madd_scores = [1.0/(scaled_madd + 1) for scaled_madd in scaled_madds]

    overall_scores = np.array(count_scores) + np.array(madd_scores)
    overall_scores = overall_scores / np.sum(overall_scores)

    sampled_point  = np.random.choice(eps_points, p=overall_scores)

    sampled_point.sampling_count += 1

    return sampled_point

def plot_frontier(hull_points:List[ConvexHullPoint], convex_hull_eps:float,
                   expdir:str)->None:
    front_points, eps_points, xs, ys = model_descs_on_front(hull_points,
        convex_hull_eps)

    # save a plot of the convex hull to aid debugging

    hull_xs = [p.model_stats.MAdd for p in eps_points]
    hull_ys = [1.0-p.metrics.best_val_top1() for p in eps_points]
    bound_xs = [p.model_stats.MAdd for p in front_points]
    bound_ys = [(1.0-p.metrics.best_val_top1()) * (1+convex_hull_eps) \
                for p in front_points]

    # for easier interpretation report everything in million increments
    xs_m = [x/1e6 for x in xs]
    hull_xs_m = [x/1e6 for x in hull_xs]
    bound_xs_m = [x/1e6 for x in bound_xs]


    plt.clf()
    plt.plot(bound_xs_m, bound_ys, c='red', label='eps-bound')
    plt.scatter(xs_m, ys, label='pts')
    plt.scatter(hull_xs_m, hull_ys, c='black', marker='+', label='eps-hull')
    plt.xlabel('Multiply-Additions (Millions)')
    plt.ylabel('Top1 Error')
    plt.savefig(os.path.join(expdir, 'convex_hull.png'),
        dpi=plt.gcf().dpi, bbox_inches='tight')

def plot_pool(hull_points:List[ConvexHullPoint], expdir:str)->None:
    assert(len(hull_points) > 0)

    xs_madd = []
    xs_flops = []
    xs_params = []
    ys = []
    for p in hull_points:
        xs_madd.append(p.model_stats.MAdd)
        xs_flops.append(p.model_stats.Flops)
        xs_params.append(p.model_stats.parameters)
        ys.append(p.metrics.best_val_top1())

    # for easier interpretation report everything in million increments
    xs_madd_m = [x/1e6 for x in xs_madd]
    xs_flops_m = [x/1e6 for x in xs_flops]
    xs_params_m = [x/1e6 for x in xs_params]

    madds_plot_filename = os.path.join(expdir, 'model_gallery_accuracy_madds.png')

    plt.clf()
    plt.scatter(xs_madd_m, ys)
    plt.xlabel('Multiply-Additions (Millions)')
    plt.ylabel('Top1 Accuracy')
    plt.savefig(madds_plot_filename, dpi=plt.gcf().dpi, bbox_inches='tight')

    flops_plot_filename = os.path.join(expdir, 'model_gallery_accuracy_flops.png')

    plt.clf()
    plt.scatter(xs_flops_m, ys)
    plt.xlabel('Flops (Millions)')
    plt.ylabel('Top1 Accuracy')
    plt.savefig(flops_plot_filename, dpi=plt.gcf().dpi, bbox_inches='tight')

    params_plot_filename = os.path.join(expdir, 'model_gallery_accuracy_params.png')

    plt.clf()
    plt.scatter(xs_params_m, ys)
    plt.xlabel('Params (Millions)')
    plt.ylabel('Top1 Accuracy')
    plt.savefig(params_plot_filename, dpi=plt.gcf().dpi, bbox_inches='tight')



def plot_seed_model_stats(seed_model_stats:List[ModelStats], expdir:str)->None:
    xs_madd = [p.MAdd for p in seed_model_stats]
    xs_madd_m = [x/1e6 for x in xs_madd]
    ys_zero = [0 for x in xs_madd]

    madds_plot_filename = os.path.join(expdir, 'seed_models_madds.png')
    plt.clf()
    plt.scatter(xs_madd_m, ys_zero)
    plt.xlabel('Multiply-Additions (Millions)')
    plt.savefig(madds_plot_filename, dpi=plt.gcf().dpi, bbox_inches='tight')


def save_hull_frontier(hull_points:List[ConvexHullPoint], convex_hull_eps:float,
               final_desc_foldername:str, expdir:str)->ConvexHullPoint:
    # make folder to save gallery of models after search
    final_desc_dir = utils.full_path(final_desc_foldername, create=True)

    # save the front on hull
    front_points, eps_points, xs, ys = model_descs_on_front(hull_points,
                                                            convex_hull_eps)
    for i, eps_point in enumerate(eps_points):
        # save readable model desc yaml
        eps_point.model_desc.save(os.path.join(final_desc_dir, f'model_desc_{i}.yaml'))
        # save hull point
        eps_point.model_desc.clear_trainables() # make file lightweight
        utils.write_string(os.path.join(final_desc_dir, f'hull_{i}.yaml'), yaml.dump(eps_point))

    front_summary_filepath = os.path.join(expdir, 'pareto_front_summary.tsv')
    utils.write_string(front_summary_filepath, hull_points2tsv(front_points))

    eps_summary_filepath = os.path.join(expdir, 'pareto_eps_summary.tsv')
    utils.write_string(eps_summary_filepath, hull_points2tsv(eps_points))

    xy_filepath = os.path.join(expdir, 'pareto_xy.tsv')
    utils.write_string(xy_filepath, '\n'.join([str(x)+'\t'+str(y) \
                                                for x,y in utils.zip_eq(xs, ys)]))
    # return last model as best performing
    return eps_points[-1]

def save_hull(hull_points:List[ConvexHullPoint], expdir:str)->None:
    full_pool_filepath = os.path.join(expdir, 'full_pool.tsv')
    utils.write_string(full_pool_filepath, hull_points2tsv(hull_points))
