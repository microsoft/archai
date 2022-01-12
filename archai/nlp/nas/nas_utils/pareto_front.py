# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Utilities that calculates the Pareto front.
"""


import copy
import os
import pickle
import re

import numpy as np
import yaml
from matplotlib import pyplot as plt

from archai.nlp.nas.constraints import get_model
from archai.nlp.nas.nas_utils.metrics import spearman_ranking
from archai.nlp.nas.nas_utils.parser import parse_results_from_experiment

model_config_defaults = {'d_head': None,
                         'n_token': 267736,
                         'dropout': 0.1,
                         'dropatt': 0.0,
                         'd_embed': None,
                         'div_val': 4,
                         'pre_lnorm': False,
                         'tgt_len': 192,
                         'ext_len': 0,
                         'mem_len': 192,
                         'same_length': False,
                         'attn_type': 0,
                         'clamp_len': -1,
                         'sample_softmax': -1,
                         'cutoffs': [19997, 39997, 199997],
                         'tie_projs': [False, True, True, True],
                         'tie_weight': True,
                         'dtype': None,
                         'primer_conv': False,
                         'primer_square': False,
                         'use_cache': False}


def get_convex_hull(xs, ys, eps=None, allow_decrease=False, allow_increase=False, results_path=None):
    """
    Andrew's Monotone Chain Algorithm: (https://en.wikipedia.org/wiki/Graham_scan)
    Assume the data are sorted in order of xs, then the computation complexity is O(n)
    If not sorted, then a sort by x-value is applied first. The complexity becomes O(nlog(n))
    Return:
    hull_indices (list): indices for the points on the hull exactly
    eps_indices (list): indices for the points on the hull + eps tolerance
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


def test_convex_hull(args):
    results_path = args['results_path']

    np.random.seed(0)
    xs = np.random.uniform(size=100)
    ys = np.random.uniform(size=100) + xs + 1.0
    eps = 0

    hull_indices, indices = get_convex_hull(xs, ys, eps, allow_decrease=False, allow_increase=True)

    hull_xs = [xs[i] for i in indices]
    hull_ys = [ys[i] for i in indices]

    bound_xs = [xs[i] for i in hull_indices]
    bound_ys = [ys[i] * (1+eps) for i in hull_indices]

    plt.plot(bound_xs, bound_ys, c='red', label='eps-bound')
    plt.scatter(xs, ys, label='pts')
    plt.scatter(hull_xs, hull_ys, c='black', marker='+', label='eps-hull')
    plt.show()
    plt.savefig(os.path.join(results_path, 'debug_convex_hull.png'), dpi=plt.gcf().dpi, bbox_inches='tight')


def test_pareto(args, eps=0.01, decreasing=True):
    np.random.seed(0)
    xs = np.random.uniform(size=1000)
    scale_x = (-1) if decreasing else (1)
    ys = np.random.uniform(size=1000) + (scale_x * xs) + 1.0

    get_vanilla_pareto(args, xs, ys, eps, decreasing)


def get_vanilla_pareto(args, xs, ys, eps=0.01, decreasing=True):
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


def get_diff_with_pareto(gt_latencies, gt_val_ppls, is_gt_pareto, is_proxy_pareto, min_acceptable_latency_diff=0):
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


def get_gt_pareto(args, alg, exp_name, path_to_dir, start_config, ppl_eps=0.1, latency_eps=0.01, hybrid=False, use_convex_hull=False,
                  min_acceptable_latency_diff=0, baseline_exp=None):
    gt_results = parse_results_from_experiment(exp_name, os.path.join(path_to_dir, exp_name), filetypes=['config.yaml', '.json'], verbose=False)
    
    print('found %d model configurations' % len(gt_results.keys()))
    print('Loading the latencies from log file')

    results_path = args['results_path']
    path_to_pkl = os.path.join(results_path, 'latencies.pkl')

    with open(path_to_pkl, 'rb') as f:
        latencies = pickle.load(f)

    print(len(latencies.keys()))

    if baseline_exp is not None:
        print('Loading the baseline')
        latencies_baseline, params_baseline, val_ppls_baseline = analyze_baseline(args, exp_name=baseline_exp, path_to_dir=path_to_dir)

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
        gene = alg.converter.config2gene(result)
        key = alg.converter.gene2key(gene)

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

        gt_pareto_indices, _ = get_convex_hull(xs, ys, eps=0., allow_decrease=True, allow_increase=False)

        is_gt_pareto[gt_pareto_indices] = 1.0
    else:
        # extract the actual pareto front based on val ppl and latency
        pareto_indices = get_vanilla_pareto(args, xs=gt_latencies, ys=gt_val_ppls, eps=latency_eps, decreasing=True)
        is_gt_pareto = np.zeros_like(is_pareto)

        for i in range(len(gt_val_ppls)):
            if i in pareto_indices:
                is_gt_pareto[i] = 1

    print('{} points on the groud-truth pareto'.format(len(np.nonzero(is_gt_pareto)[0])))

    TPR = len(np.intersect1d(np.nonzero(is_gt_pareto)[0], np.nonzero(is_pareto)[0]))*100./len(np.nonzero(is_gt_pareto)[0])
    TNR = len(np.intersect1d(np.nonzero(~is_gt_pareto)[0], np.nonzero(~is_pareto)[0]))*100./len(np.nonzero(~is_gt_pareto)[0])
    
    print(f'TPR={TPR}% and TNR={TNR}%')

    mean_ppl_difference = get_diff_with_pareto(gt_latencies, gt_val_ppls, is_gt_pareto, is_pareto, min_acceptable_latency_diff=min_acceptable_latency_diff)
    
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



def compare_w_baseline(args, alg, exp_name, path_to_dir, start_config=None, ppl_eps=0.1, latency_eps=0.01, hybrid=False, use_convex_hull=False,
                       min_acceptable_latency_diff=0, baseline_exp=None, check_pareto=True):
    gt_results = parse_results_from_experiment(exp_name, os.path.join(path_to_dir, exp_name), filetypes=['config.yaml', '.json'], verbose=False)
    
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
    latencies_baseline, params_baseline, val_ppls_baseline = analyze_baseline(args, exp_name=baseline_exp, path_to_dir=path_to_dir)

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
        gene = alg.converter.config2gene(result)
        key = alg.converter.gene2key(gene)

        if key in latencies.keys() and not key in gt_keys:
            print(job_name)
            config_number = re.search('config_([0-9]+)_', job_name).group(1)

            if int(config_number) < start_config:
                continue

            try:
                if not check_pareto or ((loaded_pareto and loaded_pareto[key]) or (not loaded_pareto and 'pareto' in job_name)):
                    model_config = copy.deepcopy(model_config_defaults)
                    model_config.update(alg.converter.gene2config(gene))
                    model = get_model(model_config)

                    params = model.get_params()
                    params_attention = params['attention']
                    params_ff = params['ff']

                    gt_val_ppls.append(result['valid_perplexity'])
                    gt_latencies.append(latencies[key])
                    gt_params.append(params_attention + params_ff)
                    gt_keys.append(key)
            except:
                pass

    gt_latencies = np.asarray(gt_latencies)
    gt_val_ppls = np.asarray(gt_val_ppls)

    print(f'found {len(gt_val_ppls)} models on the proxy pareto')

    return


def get_final_pareto_front(args, alg, eps=0.05, hybrid=False, use_convex_hull=False):
    # exract final pareto after evolutionary search has finished
    if use_convex_hull:
        print(f'extracting the pareto using the convex hull')
    else:
        print(f'extracting the pareto with eps={eps}')

    results_path = args['results_path']
    path_to_logs = os.path.join(results_path, 'logs.pkl')

    with open(path_to_logs, 'rb') as f:
        logs = pickle.load(f)

    # get all population
    seen_keys = []
    all_population = []
    all_params = []
    all_latencies = []
    idx = 0

    for i in range(len(logs['params'])):
        pop = logs['population'][i]

        if (args['start_train'] < args['n_iter']) and (i == args['start_train']) and hybrid:
            idx = len(all_population)

        for j, gene in enumerate(pop):
            key = alg.converter.gene2key(gene)
            if not key in seen_keys:
                seen_keys.append(key)
                all_population.append(gene)
                all_latencies.append(logs['latencies'][i][j])

                if hybrid:
                    all_params.append(logs['params'][i][j])   # -val_ppl

                else:
                    if (args['start_train'] < args['n_iter']) and (i >= args['start_train']):
                        model_config = copy.deepcopy(model_config_defaults)
                        model_config.update(alg.converter.gene2config(gene))
                        model = get_model(model_config, train=False)

                        params = model.get_params()
                        params_attention = params['attention']
                        params_ff = params['ff']

                        all_params.append(params_attention + params_ff)
                    else:
                        all_params.append(logs['params'][i][j])

    pareto = {'population': [],
              'params': [],
              'latencies': [],
              'keys': []}

    if not hybrid:
        assert idx == 0

    is_pareto_dict = {}
    if use_convex_hull:
        ################# pareto extraction via convex hull #################
        if hybrid:
            xs = all_latencies[idx:]
            ys = all_params[idx:]

        else:
            xs = all_params[idx:]
            ys = all_latencies[idx:]

        pareto_indices, _ = get_convex_hull(xs, ys, eps=0., allow_decrease=False)

        for i in range(len(all_params)):
            if i < idx and hybrid:
                is_pareto_dict[seen_keys[i]] = False
            else:
                is_pareto = (i in pareto_indices)
                if is_pareto:
                    pareto['population'].append(all_population[i])
                    pareto['params'].append(all_params[i])
                    pareto['latencies'].append(all_latencies[i])

                is_pareto_dict[seen_keys[i]] = is_pareto
    else:
        ################# faster vanilla pareto extraction on sorted values #################
        pareto_indices = get_vanilla_pareto(args, xs=all_params, ys=all_latencies, eps=eps, decreasing=False)
        
        for i in range(len(all_params)):
            if i in pareto_indices:
                is_pareto_dict[seen_keys[i]] = True

                pareto['params'].append(all_params[i])
                pareto['latencies'].append(all_latencies[i])
                pareto['keys'].append(seen_keys[i])

            else:
                is_pareto_dict[seen_keys[i]] = False

    fname = 'pareto{}'.format('' if hybrid else '_params')
    fname += '_convexHull' if use_convex_hull else ''

    path_to_logs = os.path.join(results_path, fname+'.pkl')
    with open(path_to_logs, 'wb') as f:
        pickle.dump(is_pareto_dict, f)

    path_to_logs = os.path.join(results_path, fname+'_points.pkl')
    with open(path_to_logs, 'wb') as f:
        pickle.dump(pareto, f)

    print(f'found {np.sum(list(is_pareto_dict.values()))} points on the proxy pareto')

    plt.figure()

    if (args['start_train'] < args['n_iter']) and hybrid:
        x = np.asarray(all_latencies[idx:]) * 1000.
        y = np.asarray(all_params[idx:]) * (-1)

        x_pareto = np.asarray(pareto['latencies']) * 1000.
        y_pareto = np.asarray(pareto['params']) * (-1)

        x_label = 'Latency (ms)'
        y_label = 'Val ppl'

    else:
        x = np.asarray(all_params)
        y = np.asarray(all_latencies) * 1000.

        x_pareto = np.asarray(pareto['params'])
        y_pareto = np.asarray(pareto['latencies']) * 1000.

        x_label = 'Decoder nParams'
        y_label = 'Latency (ms)'

    plt.scatter(x, y, s=5)
    plt.scatter(x_pareto, y_pareto, s=5, color='tab:orange')
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.ylim(0, 1000)
    plt.title('Pareto Curve')
    plt.grid(axis='y')
    plt.savefig(os.path.join(results_path, 'final_search_pareto{}.png'.format('' if hybrid else '_params')), bbox_inches="tight")




def analyze_baseline(args, exp_name, path_to_dir):
    baseline_results = parse_results_from_experiment(exp_name, os.path.join(path_to_dir, exp_name), filetypes=['config.yaml', '.json'], verbose=False)

    with open(os.path.join(path_to_dir, exp_name, 'latency_summary_{}.yaml'.format(args['device_name'])), 'r') as f:
        latencies = yaml.load(f)

    with open(os.path.join(path_to_dir, exp_name, 'params.pkl'), 'rb') as f:
        params = pickle.load(f)

    latencies_list = []
    params_list = []
    val_ppls = []

    for config_name, latency in latencies.items():
        latencies_list.append(latency)
        params_list.append(params[config_name])
        val_ppls.append(baseline_results[config_name]['valid_perplexity'])

    print(f'summarized {len(latencies.keys())} baseline jobs')

    plt.figure()
    plt.scatter(np.asarray(latencies_list) * 1000., val_ppls, s=5)
    plt.xlabel('Latency (ms)')
    plt.ylabel('Val PPL')
    plt.grid(axis='y')
    fname = 'baseline_pareto_latency.png'
    plt.savefig(os.path.join(args['results_path'], fname), bbox_inches="tight")

    return latencies_list, params_list, val_ppls
