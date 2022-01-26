# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Utilities that calculates the Pareto front.
"""

import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt

from archai.nlp.models.model_loader import load_model_from_args
from archai.nlp.nas.nas_utils.metrics import spearman_ranking
from archai.nlp.nas.nas_utils.parser import (
    parse_results_from_baseline_experiment, parse_results_from_experiment)


def calculate_convex_hull(xs: List[Any],
                          ys: List[Any],
                          eps: Optional[float] = None,
                          allow_decrease: Optional[bool] = False,
                          allow_increase: Optional[bool] = False,
                          results_path: Optional[str] = None) -> Tuple[List[Tuple], List[Tuple]]:
    """Calculates the convex hull between input points.

    Andrew's Monotone Chain Algorithm: (https://en.wikipedia.org/wiki/Graham_scan).

    Assumes the data is sorted in order of xs, then the computation complexity is O(n).
    If not sorted, then a sort by x-value is applied first, thus, complexity becomes O(nlog(n)).

    Args:
        xs: Input `x` points.
        ys: Input `y` points.
        eps: Epsilon value.
        allow_decrease: Whether convex hull should be decreased or not.
        allow_increase: Whether convex hull should be increased or not.
        results_path: Path to save the output files.

    Returns:
        (Tuple[List[Tuple], List[Tuple]]): Indices for the points on the hull and
            on the hull + epsilon.

    """
    
    xs = list(xs)
    ys = list(ys)

    indices = list(range(len(xs)))
    is_monotone = True

    for i in range(1, len(xs)):
        if xs[i] < xs[i-1]:
            is_monotone = False
            break

    if not is_monotone:
        indices.sort(key=lambda i: (xs[i], ys[i]))

    def _is_on_ray_left(x1: Any, y1: Any,
                        x2: Any, y2: Any,
                        x3: Any, y3: Any,
                        inclusive: Optional[bool] = False,
                        epsilon: Optional[float] = 0.0) -> bool:

        val = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)

        if inclusive:
            return val >= epsilon

        return val > epsilon

    def _remove_non_hull_idx(x1: Any, y1: Any, idxs: List[int]) -> List[int]:
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

    if not allow_decrease:
        xs.insert(0, xs[indices[0]]/2)
        ys.insert(0, np.min(ys).tolist())

        indices = (np.asarray(indices)+1).tolist()
        indices.insert(0, 0)

    c = 0
    min_y = float('inf')

    for idx in indices:
        x1, y1 = xs[idx], ys[idx]

        min_y = min(y1, min_y)

        hull_indices = _remove_non_hull_idx(x1, y1, hull_indices)
        hull_indices.append(idx)

        if results_path is not None:
            plt.scatter(xs, ys, label='pts', s=5)

            hull_xs = [xs[i] for i in hull_indices]
            hull_ys = [ys[i] for i in hull_indices]

            plt.scatter(hull_xs, hull_ys, c='black', label='eps-hull', s=5)
            plt.ylim((0, 1))
            plt.savefig(os.path.join(results_path, 'debug_convex_hull_{}.png'.format(c)), dpi=plt.gcf().dpi, bbox_inches='tight')
            
            c += 1

    if not allow_increase:
        # use a fake final point at (2 * x_max , y_min) to remove increasing.
        x1, y1 = xs[indices[-1]] * 2, min_y
        hull_indices = _remove_non_hull_idx(x1, y1, hull_indices)

    # compute epsilon hull (convex hull + (1+eps) band)
    eps_indices = hull_indices
    if eps is not None and eps > 0:
        eps_indices = []
        h_idx = 0  # right idx, in the hull_indices
        for idx in indices:
            x = xs[idx]
            y = ys[idx]

            if h_idx >= len(hull_indices):
                # Larger than the largest model on the hull
                y_interp = ys[hull_indices[-1]]
            elif idx == hull_indices[h_idx]:
                # critical pts on hull
                y_interp = y
                x1, y1 = x, y  # hull point to left

                h_idx += 1
                if h_idx < len(hull_indices):
                    x2, y2 = xs[hull_indices[h_idx]], ys[hull_indices[h_idx]]
                else:
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
                assert x1 <= x and x2 >= x, "idx={} idx[h_idx-1]={} idx[h_idx]={}  x={} y={} x1={} x2={} y1={} y2={} y_interp={}".format(idx, hull_indices[h_idx-1], hull_indices[h_idx], x, y, x1, x2, y1, y2, y_interp)

    if not allow_decrease:
        hull_indices.pop(0)
        hull_indices = (np.asarray(hull_indices)-1).tolist()

        eps_indices.pop(0)
        eps_indices = (np.asarray(eps_indices)-1).tolist()

    return hull_indices, eps_indices


def find_vanilla_pareto(args: Dict[str, Any],
                       xs: List[Any],
                       ys: List[Any],
                       eps: Optional[float] = 0.01,
                       decreasing: Optional[bool] = True) -> List[int]:
    """Finds the initial Pareto front.

    Args:
        args: Additional arguments.
        xs: Input `x` points.
        ys: Input `y` points.
        eps: Epsilon value.
        decreasing: Whether it is a decreasing Pareto front or not.

    Returns:
        (List[int]): Indices of Pareto front points.

    """

    results_path = args['results_path']

    range_x = np.max(xs) - np.min(xs)

    i = 0
    pareto_indices = []
    for i in range(len(xs)):
        curr_x = xs[i]

        # find a range where the change on x axis is smaller than eps %
        indices = []
        for j in range(len(xs)):
            diff_x = np.absolute(xs[j] - curr_x)

            if diff_x <= eps*range_x:
                indices.append(j)

        # mark the pareto front point in the found range
        curr_ys = [ys[k] for k in indices]
        pareto_idx = indices[np.argmin(curr_ys)]

        if pareto_idx not in pareto_indices:
            pareto_indices.append(pareto_idx)

    print(f'initially found {len(pareto_indices)} pareto points')

    to_remove = []
    for i in pareto_indices:
        for j in pareto_indices:
            if j == i:
                continue
            if decreasing:
                if xs[i] >= xs[j] and ys[i] >= ys[j]:
                    to_remove.append(i)
                    break
            else:
                if xs[i] < xs[j] and ys[i] >= ys[j]:
                    to_remove.append(i)
                    break

    print(f'removing {len(to_remove)} non-pareto points')

    pareto_indices_pruned = [i for i in pareto_indices if i not in to_remove]

    pareto_xs = [xs[i] for i in pareto_indices_pruned]
    pareto_ys = [ys[i] for i in pareto_indices_pruned]

    plt.scatter(xs, ys, s=5)
    plt.scatter(pareto_xs, pareto_ys, label='pareto', s=5)
    plt.scatter([xs[i] for i in to_remove], [ys[i] for i in to_remove], c='black', marker='+', label='removed', s=5)
    plt.legend()
    plt.ylim((0, 1))
    plt.savefig(os.path.join(results_path, 'debug_pareto_front.png'), dpi=plt.gcf().dpi, bbox_inches='tight')

    return pareto_indices_pruned


def find_diff_between_paretos(gt_latencies: List[float],
                              gt_val_ppls: List[float],
                              is_gt_pareto: List[bool],
                              is_proxy_pareto: List[bool],
                              min_acceptable_latency_diff: Optional[float] = 0.0) -> float:
    """Finds difference between ground-truth Pareto front and found Pareto front.

    Args:
        gt_latencies: Ground-truth latencies.
        gt_val_ppls: Ground-truth validation perplexities.
        is_gt_pareto: Whether point is in ground-truth Pareto or not.
        is_proxy_pareto: Whether point is in proxy Pareto or not.
        min_acceptable_latency_diff: Minimum acceptable latency difference.

    Returns:
        (float): Perplexity difference between ground-truth and found Paretos.

    """

    sorted_idx = np.argsort(gt_latencies)

    ppl_diff = []
    on_pareto = 0

    for i, idx in enumerate(sorted_idx):
        latency, val_ppl = gt_latencies[idx], gt_val_ppls[idx]

        if not is_proxy_pareto[idx]:
            continue

        if is_gt_pareto[idx]:
            ppl_diff.append(0.)
            on_pareto += 1
        else:
            idx_fwd, idx_bkwd = None, None

            for j in sorted_idx[i+1:]:
                if is_gt_pareto[j]:
                    diff_frwd = np.absolute(latency-gt_latencies[j])
                    idx_fwd = j
                    break

            idx_range = sorted_idx[0:i][::-1]
            for j in idx_range:
                if is_gt_pareto[j]:
                    diff_bkwd = np.absolute(latency-gt_latencies[j])
                    idx_bkwd = j
                    break

            if idx_fwd is None:
                closest_idx = idx_bkwd
            elif idx_bkwd is None or diff_frwd < diff_bkwd:
                closest_idx = idx_fwd
            else:
                closest_idx = idx_bkwd

            latency_diff = np.absolute(latency-gt_latencies[closest_idx])*1000
            print('latency difference with closest pareto point: {:1f} ms'.format(latency_diff))

            if latency_diff <= min_acceptable_latency_diff:
                ppl_diff.append(np.absolute(val_ppl-gt_val_ppls[closest_idx])*100./gt_val_ppls[closest_idx])

    if min_acceptable_latency_diff == 0:
        assert len(ppl_diff) == np.sum(is_proxy_pareto)

    print(f'{on_pareto} points out of {np.sum(is_proxy_pareto)} were on the ground-truth pareto')

    return np.mean(ppl_diff)


def find_ground_truth_pareto(args: Dict[str, Any],
                             alg: Any,
                             exp_name: str,
                             path_to_dir: str,
                             start_config: int,
                             ppl_eps: Optional[float] = 0.1,
                             latency_eps: Optional[float] = 0.01,
                             hybrid: Optional[bool] = False,
                             use_convex_hull: Optional[bool] = False,
                             min_acceptable_latency_diff: Optional[float] = 0.0,
                             baseline_exp: Optional[str] = None) -> None:
    """Find the ground-truth Pareto front.

    Args:
        args: Additional arguments.
        alg: Evolutionary algorithm.
        exp_name: Name of experiment.
        path_to_dir: Path to the experiment's files directory.
        start_config: Starting range of the configuration to be checked.
        ppl_eps: Perplexity epsilon.
        latency_eps: Latency epsilon.
        hybrid: Whether is a hybrid Pareto front or not.
        use_convex_hull: Whether convex hull calculation should be used or not.
        min_acceptable_latency_diff: Minimum acceptable latency difference.
        baseline_exp: Baseline experiment name.

    """

    gt_results = parse_results_from_experiment(exp_name, os.path.join(path_to_dir, exp_name), file_type=['config.yaml', '.json'], verbose=False)
    
    print('found %d model configurations' % len(gt_results.keys()))
    print('Loading the latencies from log file')

    results_path = args['results_path']
    path_to_pkl = os.path.join(results_path, 'latencies.pkl')

    with open(path_to_pkl, 'rb') as f:
        latencies = pickle.load(f)

    print(len(latencies.keys()))

    if baseline_exp is not None:
        print('Loading the baseline')
        latencies_baseline, params_baseline, val_ppls_baseline = parse_results_from_baseline_experiment(args, exp_name=baseline_exp, path_to_dir=path_to_dir)

    # load previous pareto
    loaded_pareto = None
    fname = 'pareto{}'.format('' if hybrid else '_params')
    fname += '_convexHull' if use_convex_hull else ''
    path_to_pkl = os.path.join(results_path, fname+'.pkl')

    if os.path.exists(path_to_pkl):
        print('Loading proxy pareto')
        with open(path_to_pkl, 'rb') as f:
            loaded_pareto = pickle.load(f)

    gt_latencies = []
    gt_val_ppls = []
    is_pareto = []
    gt_keys = []

    for job_name, result in gt_results.items():
        gene = alg.converter.config_to_gene(result)
        key = alg.converter.gene_to_str(gene)

        if key in latencies.keys() and not key in gt_keys:
            config_number = re.search('config_([0-9]+)_', job_name).group(1)

            if int(config_number) < start_config:
                continue

            try:
                gt_val_ppls.append(result['valid_perplexity'])
                gt_latencies.append(latencies[key])

                if loaded_pareto:
                    is_pareto.append(loaded_pareto[key])
                else:
                    is_pareto.append(True if 'pareto' in job_name else False)

                gt_keys.append(key)
            except:
                pass

    is_pareto = np.asarray(is_pareto)
    print(f'found {len(gt_val_ppls)} models with {np.sum(is_pareto)} on the proxy pareto')
    
    if use_convex_hull:
        ################# pareto extraction via convex hull #################
        assert len(gt_val_ppls) == len(is_pareto)
        is_gt_pareto = np.zeros_like(is_pareto)

        xs = gt_latencies
        ys = np.asarray(gt_val_ppls)

        gt_pareto_indices, _ = calculate_convex_hull(xs, ys, eps=0., allow_decrease=True, allow_increase=False)

        is_gt_pareto[gt_pareto_indices] = 1.0
    else:
        # extract the actual pareto front based on val ppl and latency
        pareto_indices = find_vanilla_pareto(args, xs=gt_latencies, ys=gt_val_ppls, eps=latency_eps, decreasing=True)
        is_gt_pareto = np.zeros_like(is_pareto)

        for i in range(len(gt_val_ppls)):
            if i in pareto_indices:
                is_gt_pareto[i] = 1

    print('{} points on the groud-truth pareto'.format(len(np.nonzero(is_gt_pareto)[0])))

    TPR = len(np.intersect1d(np.nonzero(is_gt_pareto)[0], np.nonzero(is_pareto)[0]))*100./len(np.nonzero(is_gt_pareto)[0])
    TNR = len(np.intersect1d(np.nonzero(~is_gt_pareto)[0], np.nonzero(~is_pareto)[0]))*100./len(np.nonzero(~is_gt_pareto)[0])
    
    print(f'TPR={TPR}% and TNR={TNR}%')

    mean_ppl_difference = find_diff_between_paretos(gt_latencies, gt_val_ppls, is_gt_pareto, is_pareto, min_acceptable_latency_diff=min_acceptable_latency_diff)
    
    print('mean ppl difference between proxy and gt pareto: {:.1f}%'.format(mean_ppl_difference))

    plt.figure(figsize=(5, 3))
    plt.scatter(np.asarray(gt_latencies)[~is_pareto] * 1000., np.asarray(gt_val_ppls)[~is_pareto], s=5, label='Ground-truth', color='midnightblue')
    plt.scatter(np.asarray(gt_latencies)[is_pareto] * 1000., np.asarray(gt_val_ppls)[is_pareto], s=5, label='Proxy pareto', color='tab:orange')
    
    if baseline_exp:
        plt.scatter(np.asarray(latencies_baseline) * 1000., np.asarray(val_ppls_baseline), s=25, marker='*', c='red', label='Baseline')
        plt.xlim((min(np.min(gt_latencies), np.min(latencies_baseline)) * 1000-10, np.max(gt_latencies)*1000+10))
    else:
        plt.xlim(np.min(gt_latencies)*1000-10, np.max(gt_latencies)*1000+10)

    plt.xlabel('Latency (ms)')
    plt.ylabel('Val PPL')
    plt.grid(axis='y')
    plt.legend(handletextpad=0.1, borderpad=0)
    fname = 'gt_pareto_latency{}.png'.format('' if hybrid else '_params')
    plt.savefig(os.path.join(results_path, fname), bbox_inches="tight")


def compare_pareto_fronts(args: Dict[str, Any],
                          alg: Any,
                          exp_name: str,
                          path_to_dir: str,
                          start_config: int,
                          ppl_eps: Optional[float] = 0.1,
                          latency_eps: Optional[float] = 0.01,
                          hybrid: Optional[bool] = False,
                          use_convex_hull: Optional[bool] = False,
                          min_acceptable_latency_diff: Optional[float] = 0.0,
                          baseline_exp: Optional[str] = None,
                          check_pareto: Optional[bool] = True) -> None:
    """Compare Pareto fronts (baseline and proxy).

    Args:
        args: Additional arguments.
        alg: Evolutionary algorithm.
        exp_name: Name of experiment.
        path_to_dir: Path to the experiment's files directory.
        start_config: Starting range of the configuration to be checked.
        ppl_eps: Perplexity epsilon.
        latency_eps: Latency epsilon.
        hybrid: Whether is a hybrid Pareto front or not.
        use_convex_hull: Whether convex hull calculation should be used or not.
        min_acceptable_latency_diff: Minimum acceptable latency difference.
        baseline_exp: Baseline experiment name.
        check_pareto: Whether Pareto front should be checked or not.

    """

    gt_results = parse_results_from_experiment(exp_name, os.path.join(path_to_dir, exp_name), file_type=['config.yaml', '.json'], verbose=False)
    
    print('found %d model configurations' % len(gt_results.keys()))
    print('Loading the latencies from log file')

    results_path = args['results_path']
    path_to_pkl = os.path.join(results_path, 'latencies.pkl')

    if not os.path.exists(path_to_pkl):
        path_to_pkl = os.path.join(results_path, 'pareto_latencies.pkl')

    with open(path_to_pkl, 'rb') as f:
        latencies = pickle.load(f)

    assert baseline_exp is not None, 'please provide a baseline'

    print('Loading the baseline')
    latencies_baseline, params_baseline, val_ppls_baseline = parse_results_from_baseline_experiment(args, exp_name=baseline_exp, path_to_dir=path_to_dir)

    sorted_params_baseline = np.argsort(params_baseline)
    sorted_val_ppls_baseline = np.argsort(val_ppls_baseline)

    print('################# baseline')
    common_ratio, spr_rank = spearman_ranking(100, sorted_val_ppls_baseline, sorted_target=sorted_params_baseline,
                                         val_ppl_list_gt=val_ppls_baseline, val_ppl_list_target=params_baseline)

    # load previous pareto
    loaded_pareto = None
    fname = 'pareto{}'.format('' if hybrid else '_params')
    fname += '_convexHull' if use_convex_hull else ''
    path_to_pkl = os.path.join(results_path, fname+'.pkl')

    if os.path.exists(path_to_pkl):
        print('Loading proxy pareto')
        with open(path_to_pkl, 'rb') as f:
            loaded_pareto = pickle.load(f)

    gt_latencies = []
    gt_params = []
    gt_val_ppls = []
    gt_keys = []

    for job_name, result in gt_results.items():
        gene = alg.converter.config_to_gene(result)
        key = alg.converter.gene_to_str(gene)

        if key in latencies.keys() and not key in gt_keys:
            print(job_name)
            config_number = re.search('config_([0-9]+)_', job_name).group(1)

            if int(config_number) < start_config:
                continue

            try:
                if not check_pareto or ((loaded_pareto and loaded_pareto[key]) or (not loaded_pareto and 'pareto' in job_name)):
                    model_config = load_model_from_args('hf_gpt2', cls_type='config')
                    model_config.update(alg.converter.gene_to_config(gene))
                    model = load_model_from_args('hf_gpt2', **model_config)

                    n_params = model.get_params()
                    n_params_attention = n_params['attention']
                    n_params_ff = n_params['ff']

                    gt_val_ppls.append(result['valid_perplexity'])
                    gt_latencies.append(latencies[key])
                    gt_params.append(n_params_attention + n_params_ff)
                    gt_keys.append(key)
            except:
                pass

    gt_latencies = np.asarray(gt_latencies)
    gt_val_ppls = np.asarray(gt_val_ppls)

    print(f'found {len(gt_val_ppls)} models on the proxy pareto')




def find_pareto_points(all_points:np.array,
                    is_decreasing:bool=True)->List[int]:
    '''Takes in a list of n-dimensional points, 
    one per row, returns the list of row indices
    which are pareto-frontier points. Assumes that 
    lower values on every dimension are better.'''

    # for each point see if there exists 
    # any other point which dominates it on all dimensions
    # if that is true, then it is not a pareto point
    # and vice-versa.

    # input should be two dimensional array
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
            if sum(diff>0) == dim:
                # other point is smaller/larger on all dimensions
                # so we have found at least one dominating point
                is_pareto = False
        if is_pareto:
            pareto_inds.append(i)

    return pareto_inds



















