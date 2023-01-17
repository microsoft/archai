# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import pdb
from collections import defaultdict
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt
import seaborn as sns
import math as ma
import h5py
import os
from copy import deepcopy
from typing import List, Set, Dict, Tuple, Any, Callable
from tqdm import tqdm
from itertools import permutations, combinations

from archai.algos.divnas.seqopt import SeqOpt


def create_submod_f(covariance:np.array)->Callable:
    def compute_marginal_gain_func(item:int, sub_sel:List[int], S:Set[int]):
        assert covariance.shape[0] == covariance.shape[1]
        assert len(covariance.shape) == 2
        assert len(S) == covariance.shape[0]

        sel_set = set(sub_sel)
        marg_gain = compute_marginal_gain(item, sel_set, S, covariance)
        return marg_gain
    return compute_marginal_gain_func


def get_batch(feature_list, batch_size, i):
    start_row = batch_size * i
    end_row = start_row + batch_size
    feats = [feat[start_row:end_row, :] for feat in feature_list]
    return feats


def rbf(x:np.array, y:np.array, sigma=0.1)->np.array:
    """ Computes the rbf kernel between two input vectors """

    # make sure that inputs are vectors
    assert len(x.shape) == 1
    assert len(y.shape) == 1

    sq_euclidean = np.sum(np.square(x-y))
    k = np.exp(-sq_euclidean/(2*sigma*sigma))
    return k


def _compute_mi(cov_kernel:np.array, A:Set, V_minus_A:Set):
    sigma_A = cov_kernel[np.ix_(list(A), list(A))]
    sigma_V_minus_A = cov_kernel[np.ix_(list(V_minus_A), list(V_minus_A))]
    I = 0.5 * np.log(np.linalg.det(sigma_A) * np.linalg.det(sigma_V_minus_A) / np.linalg.det(cov_kernel))
    return I


def compute_brute_force_sol(cov_kernel:np.array, budget:int)->Tuple[Tuple[Any], float]:

    assert cov_kernel.shape[0] == cov_kernel.shape[1]
    assert len(cov_kernel.shape) == 2
    assert budget > 0 and budget <= cov_kernel.shape[0]

    V = set(range(cov_kernel.shape[0]))
    
    # for each combination of budgeted items compute its mutual
    # information with the complement set
    mis = []
    for subset in combinations(range(cov_kernel.shape[0]), budget):
        A = set(subset)
        V_minus_A = V - A
        I = _compute_mi(cov_kernel, A, V_minus_A)
        mis.append((subset, I))

    # find the maximum subset
    max_subset, mi = max(mis, key = lambda x: x[1])
    return max_subset, mi



def compute_correlation(covariance:np.array)->np.array:
    variance = np.diag(covariance).reshape(-1, 1)
    stds = np.sqrt(np.matmul(variance, variance.T))
    correlation = covariance / (stds + 1e-16)
    return correlation


def compute_covariance_offline(feature_list:List[np.array])->np.array:
    """Compute covariance matrix for high-dimensional features.
    feature_shape: (num_samples, feature_dim)
    """
    num_features = len(feature_list)
    num_samples = feature_list[0].shape[0]
    flatten_features = [
        feas.reshape(num_samples, -1) for feas in feature_list]
    unbiased_features = [
        feas - np.mean(feas, 0) for feas in flatten_features]
    # (num_samples, feature_dim, num_features)
    features = np.stack(unbiased_features, -1)
    covariance = np.zeros((num_features, num_features), np.float32)
    for i in range(num_samples):
        covariance += np.matmul(features[i].T, features[i])
    return covariance


def compute_rbf_kernel_covariance(feature_list:List[np.array], sigma=0.1)->np.array:
    """ Compute rbf kernel covariance for high dimensional features. 
    feature_list: List of features each of shape: (num_samples, feature_dim)
    sigma: sigma of the rbf kernel """
    num_features = len(feature_list)
    covariance = np.zeros((num_features, num_features), np.float32)

    for i in range(num_features):
        for j in range(num_features):
            if i == j:
                covariance[i][j] = covariance[j][i] = 1.0
                continue

            # NOTE: one could try to take all pairs rbf responses
            # but that is too much computation and probably does 
            # not add much information
            feats_i = feature_list[i]
            feats_j = feature_list[j]
            assert feats_i.shape == feats_j.shape
                        
            rbfs = np.exp(-np.sum(np.square(feats_i - feats_j), axis=1) / (2*sigma*sigma))
            avg_cov = np.sum(rbfs)/feats_i.shape[0]
            covariance[i][j] = covariance[j][i] = avg_cov

    return covariance

    
def compute_euclidean_dist_quantiles(feature_list:List[np.array], subsamplefactor=1)->List[Tuple[float, float]]:
    """ Compute quantile distances between feature pairs 
    feature_list: List of features each of shape: (num_samples, feature_dim)
    """
    num_features = len(feature_list)
    num_samples = feature_list[0].shape[0]
    # (num_samples, feature_dim, num_features)
    features = np.stack(feature_list, -1)

    # compute all pairwise feature distances
    # too slow need to vectorize asap
    distances = []
    for i in range(num_features):
        for j in range(num_features):
            if i == j:                
                continue

            for k in range(0, num_samples, subsamplefactor):
                feat_i = features[k, :][:, i]
                feat_j = features[k, :][:, j]
                dist = np.sqrt(np.sum(np.square(feat_i-feat_j)))
                distances.append(dist)

    quantiles = [i*0.1 for i in range(1, 10)]
    quant_vals = np.quantile(distances, quantiles)
    quants = []
    for quant, val in zip(quantiles, quant_vals.tolist()):
        quants.append((quant, val))
    return quants


def greedy_op_selection(covariance:np.array, k:int)->List[int]:
    assert covariance.shape[0] == covariance.shape[1]
    assert len(covariance.shape) == 2
    assert k <= covariance.shape[0]

    A = set()
    # to keep order information
    A_list = []

    S = set()
    for i in range(covariance.shape[0]):
        S.add(i)
    
    for i in tqdm(range(k)):
        marginal_gains = []
        marginal_gain_ids = []
        for y in S - A:
            delta_y = compute_marginal_gain(y, A, S, covariance)
            marginal_gains.append(delta_y)
            marginal_gain_ids.append(y)

        val = -ma.inf
        argmax = -1
        for marg_gain, marg_gain_id in zip(marginal_gains, marginal_gain_ids):
            if marg_gain > val:
                val = marg_gain
                argmax = marg_gain_id
        
        A.add(argmax)
        A_list.append(argmax)

    return A_list


def compute_marginal_gain(y:int, A:Set[int], S:Set[int], covariance:np.array)->float:

    if A:
        A_copy = deepcopy(A)
        A_copy.add(y)
    else:
        A_copy = set()
        A_copy.add(y)

    A_bar = S - A_copy

    sigma_y_sqr = covariance[y, y]

    if A:
        sigma_AA = covariance[np.ix_(list(A), list(A))]
        sigma_yA = covariance[np.ix_([y], list(A))]
        numerator = sigma_y_sqr - np.matmul(sigma_yA, np.matmul(np.linalg.inv(sigma_AA), sigma_yA.T))
    else:
        numerator = sigma_y_sqr

    if A_bar:
        sigma_AA_bar = covariance[np.ix_(list(A_bar), list(A_bar))]
        sigma_yA_bar = covariance[np.ix_([y], list(A_bar))]
        denominator = sigma_y_sqr - np.matmul(sigma_yA_bar, np.matmul(np.linalg.inv(sigma_AA_bar), sigma_yA_bar.T))
    else:
        denominator = sigma_y_sqr

    gain = numerator/denominator
    return float(gain)


def collect_features(rootfolder:str, subsampling_factor:int = 1)->Dict[str, List[np.array]]:
    """ Walks the rootfolder for h5py files and loads them into the format
    required for analysis.
    
    Inputs:

    rootfolder: full path to folder containing h5 files which have activations
    subsampling_factor: every nth minibatch will be loaded to keep memory manageable

    Outputs:

    dictionary with edge name strings as keys and values are lists of np.array [num_samples, feature_dim]
     """

    assert subsampling_factor > 0

    # gather all h5 files 
    h5files = [os.path.join(rootfolder, f) for f in os.listdir(rootfolder) if os.path.isfile(os.path.join(rootfolder, f)) and '.h5' in f]
    assert h5files


    # storage for holding activations for all edges 
    all_edges_activs = defaultdict(list)

    for h5file in h5files:
        with h5py.File(h5file, 'r') as hf:
            edge_name = h5file.split('/')[-1].split('.')[-2]
            edge_activ_list = []

            # load all batches
            keys_list = list(hf.keys())
            print(f'processing {h5file}, num batches {len(keys_list)}')
            for i in range(0, len(keys_list), subsampling_factor):
                key = keys_list[i]
                payload = np.array(hf.get(key))
                edge_activ_list.append(payload)

            obsv_dict = defaultdict(list)
            # separate activations by ops
            for batch in edge_activ_list:
                # assumption (num_ops, batch_size, x, y, z)
                for op in range(batch.shape[0]):
                    for b in range(batch.shape[1]):
                        feat = batch[op][b]
                        feat = feat.flatten()
                        obsv_dict[op].append(feat)

            num_ops = edge_activ_list[0].shape[0]
            feature_list = [np.zeros(1) for _ in range(num_ops)]
            for key in obsv_dict.keys():
                feat = np.array(obsv_dict[key])
                feature_list[key] = feat

            # removing none and skip_connect
            del feature_list[-1]
            del feature_list[2] 
        
            all_edges_activs[edge_name] = feature_list

    return all_edges_activs


def plot_all_covs(covs_kernel, corr, primitives, axs):
    assert axs.shape[0] * axs.shape[1] == len(covs_kernel) + 1
    flat_axs = axs.flatten()

    for i, quantile in enumerate(covs_kernel.keys()):
        cov = covs_kernel[quantile]
        sns.heatmap(cov, annot=True, fmt='.1g', cmap='coolwarm', xticklabels=primitives, yticklabels=primitives, ax=flat_axs[i])
        flat_axs[i].set_title(f'Kernel covariance sigma={quantile} quantile')

    sns.heatmap(corr, annot=True, fmt='.1g', cmap='coolwarm', xticklabels=primitives, yticklabels=primitives, ax=flat_axs[-1])
    flat_axs[-1].set_title(f'Correlation')

def main():

    rootfile = '/media/dedey/DATADRIVE1/activations'

    all_edges_activs = collect_features(rootfile, subsampling_factor=5)

    PRIMITIVES = [
        'max_pool_3x3',
        'avg_pool_3x3',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5',        
    ]

    # # Use all edges
    # all_edges_list = []
    # all_names_list = []
    # for i in all_edges_activs.keys():
    #     all_edges_list.extend(all_edges_activs[i])
    #     for prim in PRIMITIVES:
    #         all_names_list.append(i + '_' + prim) 

    # Use specific edges
    all_edges_list = []
    all_names_list = []

    # edge_list = ['activations_node_0_edge_0']
    edge_list = ['activations_node_0_edge_0', 'activations_node_0_edge_1']
    # edge_list = ['activations_node_1_edge_0', 'activations_node_1_edge_1', 'activations_node_1_edge_2']
    # edge_list = ['activations_node_2_edge_0', 'activations_node_2_edge_1', 'activations_node_2_edge_2', 'activations_node_2_edge_3']
    # edge_list = ['activations_node_3_edge_0', 'activations_node_3_edge_1', 'activations_node_3_edge_2', 'activations_node_3_edge_3', 'activations_node_3_edge_4']
    for name in edge_list:
        all_edges_list.extend(all_edges_activs[name])
        for prim in PRIMITIVES:
            all_names_list.append(name + '_' + prim) 

        
    # compute covariance like usual
    # cov = compute_covariance_offline(all_edges_list)
    # corr = compute_correlation(cov)
    # sns.heatmap(corr, annot=False, xticklabels=all_names_list, yticklabels=all_names_list, cmap='coolwarm')
    # plt.axis('equal')
    # plt.show()

    # compute kernel covariance
    # quants = compute_euclidean_dist_quantiles(all_edges_list, subsamplefactor=20)
    cov_kernel_orig = compute_rbf_kernel_covariance(all_edges_list, sigma=168)
    cov_kernel = cov_kernel_orig + 1.0*np.eye(cov_kernel_orig.shape[0])
    print(f'Det before diag addition {np.linalg.det(cov_kernel_orig)}')
    print(f'Det after diag addition {np.linalg.det(cov_kernel)}')
    print(f'Condition number is {np.linalg.cond(cov_kernel)}')
    sns.heatmap(cov_kernel, annot=False, xticklabels=all_names_list, yticklabels=all_names_list, cmap='coolwarm')
    plt.axis('equal')
    plt.show()

    # brute force solution
    budget = 4
    bf_sensors, bf_val = compute_brute_force_sol(cov_kernel_orig, budget)
    print(f'Brute force max subset {bf_sensors}, max mi {bf_val}')

    # greedy
    print('Greedy selection')
    greedy_ops = greedy_op_selection(cov_kernel, cov_kernel.shape[0])

    for i, op_index in enumerate(greedy_ops):
        print(f'Greedy op {i} is {all_names_list[op_index]}')

    greedy_budget = greedy_ops[:budget]
     # find MI of the greedy solution
    V = set(range(cov_kernel.shape[0]))
    A_greedy = set(greedy_budget)
    V_minus_A_greedy = V - A_greedy
    I_greedy = _compute_mi(cov_kernel_orig, A_greedy, V_minus_A_greedy)
    print(f'Greedy solution is {greedy_budget}, mi is {I_greedy}')

    # seqopt
    # simulated batch size
    batch_size = 64
    num_batches = int(all_edges_list[0].shape[0] / batch_size)

    # seqopt object that will get updated in an online manner
    num_items = cov_kernel.shape[0]
    eps = 0.1
    seqopt = SeqOpt(num_items, eps)

    for i in tqdm(range(num_batches)):
        # simulate getting a new batch of activations
        sample = get_batch(all_edges_list, batch_size, i)

        # sample a list of activations from seqopt
        sel_list = seqopt.sample_sequence(with_replacement=False)

        # Using 50th percentile distance
        sigma = 168.0
        cov = compute_rbf_kernel_covariance(sample, sigma=sigma)

        # update seqopt
        compute_marginal_gain_func = create_submod_f(cov)
        seqopt.update(sel_list, compute_marginal_gain_func)

    # now sample a list of ops and hope it is diverse
    sel_list = seqopt.sample_sequence(with_replacement=False)
    # sel_primitives = [all_names_list for i in sel_list]
    # print(f'SeqOpt selected primitives are {sel_primitives}')

    # check that it is close to greedy and or bruteforce
    budget = 4
    sel_list = sel_list[:budget]
     # find MI of the greedy solution
    V = set(range(num_items))
    A_seqopt = set(sel_list)
    V_minus_A_seqopt = V - A_seqopt
    I_seqopt = _compute_mi(cov_kernel_orig, A_seqopt, V_minus_A_seqopt)
    print(f'SeqOpt solution is {sel_list}, mi is {I_seqopt}')


    # # For enumerating through many choices of rbf sigmas
    # covs_kernel = {}
    # for quantile, val in quants:
    #     print(f'Computing kernel covariance for quantile {quantile}')
    #     cov_kernel = compute_rbf_kernel_covariance(all_edges_list, sigma=val)
    #     covs_kernel[quantile] = cov_kernel
    
    # # compute greedy sequence of ops on one of the kernels
    # print('Greedy selection')
    # greedy_ops = greedy_op_selection(covs_kernel[0.5], 3)

    # for i, op_index in enumerate(greedy_ops):
    #     print(f'Greedy op {i} is {all_names_list[op_index]}')


    # fig, axs = plt.subplots(5, 2)
    # plot_all_covs(covs_kernel, corr, all_names_list, axs)
    # plt.show()


        












if __name__ == '__main__':
    main()