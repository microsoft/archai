# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from copy import deepcopy
import math as ma
import unittest
from tqdm import tqdm
from typing import Any, Callable, List, Tuple, Set

import archai.algos.divnas.analyse_activations as aa
from archai.supergraph.algos.divnas.seqopt import SeqOpt
from archai.supergraph.algos.divnas.analyse_activations import _compute_mi, compute_brute_force_sol
from archai.supergraph.algos.divnas.analyse_activations import create_submod_f
from archai.supergraph.algos.divnas.wmr import Wmr

def create_rbf_func(first:np.array, sigma:float)->Callable:
    assert len(first.shape) == 1
    assert sigma >= 0.0
    def rbf_bound(second:np.array):
        assert len(second.shape) == 1
        val =  aa.rbf(first, second, sigma)
        return val
    return rbf_bound


def synthetic_data2()->List[Tuple[np.array, np.array]]:
    # num grid locations
    num_loc = 10
    # plop some kernels on 0, 3, 9
    k_0_func = create_rbf_func(np.array([0.0]), 3.0)
    k_3_func = create_rbf_func(np.array([3.0]), 0.1)
    k_6_func = create_rbf_func(np.array([6.0]), 0.1)
    k_9_func = create_rbf_func(np.array([9.0]), 0.5)

    Y = []
    for i in range(num_loc):
        i_arr = np.array([i])
        y = 20.0 * k_0_func(i_arr) - 25.0 * k_3_func(i_arr) + 100.0 * k_6_func(i_arr) - 100.0 * k_9_func(i_arr)
        y_arr = np.array([y])
        Y.append((y_arr, i_arr))

    return Y


def synthetic_data()->List[Tuple[np.array, np.array]]:
    # num grid locations
    num_loc = 10
    # plop some kernels on 0, 3, 9
    k_0_func = create_rbf_func(np.array([0.0]), 3.0)
    k_3_func = create_rbf_func(np.array([3.0]), 0.1)
    k_6_func = create_rbf_func(np.array([6.0]), 0.1)
    k_9_func = create_rbf_func(np.array([9.0]), 0.5)

    Y = []
    for i in range(num_loc):
        i_arr = np.array([i])
        y = -10.0 * k_0_func(i_arr) + 25.0 * k_3_func(i_arr) - 100.0 * k_6_func(i_arr) - 100.0 * k_9_func(i_arr)
        y_arr = np.array([y])
        Y.append((y_arr, i_arr))

    return Y


def compute_synthetic_data_covariance(Y:List[Tuple[np.array, np.array]], sigma=0.8):
    num_obsvs = len(Y)
    covariance = np.zeros((num_obsvs, num_obsvs), np.float32)

    for i in range(num_obsvs):
        for j in range(num_obsvs):
            if i == j:
                covariance[i][j] = covariance[j][i] = 1.0
                continue

            obsv_i = Y[i][0]
            obsv_j = Y[j][0]
            assert obsv_i.shape == obsv_j.shape
            if len(obsv_i.shape) == 1:
                obsv_i = np.reshape(obsv_i, (obsv_i.shape[0], 1))
                obsv_j = np.reshape(obsv_j, (obsv_j.shape[0], 1))

            rbfs = np.exp(-np.sum(np.square(obsv_i - obsv_j), axis=1) / (2*sigma*sigma))
            avg_cov = np.sum(rbfs)/obsv_i.shape[0]
            covariance[i][j] = covariance[j][i] = avg_cov

    return covariance




class SeqOptSyntheticDataTestCase(unittest.TestCase):

    def setUp(self):
        self.Y = synthetic_data2()
        self.vals = [item[0] for item in self.Y]
        self.cov_kernel = compute_synthetic_data_covariance(self.Y)

    def test_marginal_gain_calculation(self):
        """ Tests that marginal gain calculation is correct """
        V = set(range(self.cov_kernel.shape[0]))
        A_random = set([1])
        V_minus_A_random = V - A_random
        y = 2
        I_A_random = _compute_mi(self.cov_kernel, A_random, V_minus_A_random)
        A_aug = deepcopy(A_random)
        A_aug.add(y)
        V_minus_A_aug = V - A_aug
        I_A_aug = _compute_mi(self.cov_kernel, A_aug, V_minus_A_aug)
        diff_via_direct = abs(I_A_aug - I_A_random)
        print(f'MI(A) {I_A_random}, MI(A U y) {I_A_aug}, diff {diff_via_direct}')

        diff = aa.compute_marginal_gain(y, A_random, V, self.cov_kernel)
        # the marginal_gain leaves out 0.5 * log term as it does not
        # matter for ranking elements
        half_log_diff = 0.5 * np.log(diff)
        print(f'Diff via aa.compute {half_log_diff}')
        self.assertAlmostEqual(diff_via_direct, half_log_diff, delta=0.01)

    def test_greedy(self):
        # budgeted number of sensors
        budget = 4

        # brute force solution
        bf_sensors, bf_val = compute_brute_force_sol(self.cov_kernel, budget)
        print(f'Brute force max subset {bf_sensors}, max mi {bf_val}')

        # greedy
        greedy_sensors = aa.greedy_op_selection(self.cov_kernel, budget)
        # find MI of the greedy solution
        V = set(range(self.cov_kernel.shape[0]))
        A_greedy = set(greedy_sensors)
        V_minus_A_greedy = V - A_greedy
        I_greedy = _compute_mi(self.cov_kernel, A_greedy, V_minus_A_greedy)
        print(f'Greedy solution is {greedy_sensors}, mi is {I_greedy}')

        self.assertAlmostEqual(bf_val, I_greedy, delta=0.1)

    def test_wmr(self):
        eta = 0.01
        num_rounds = 10000
        gt_distrib = [0.15, 0.5, 0.3, 0.05]
        num_items = len(gt_distrib)
        wmr = Wmr(num_items, eta)

        for _ in range(num_rounds):
            sampled_index = np.random.choice(num_items, p=gt_distrib)
            rewards = np.zeros((num_items))
            rewards[sampled_index] = 1.0
            wmr.update(rewards)

        print(wmr.weights)
        self.assertTrue(wmr.weights[1] > 0.4)

    def test_seqopt(self):

        # budgeted number of sensors
        budget = 4

        # brute force solution
        bf_sensors, bf_val = compute_brute_force_sol(self.cov_kernel, budget)
        print(f'Brute force max subset {bf_sensors}, max mi {bf_val}')

        # greedy
        greedy_sensors = aa.greedy_op_selection(self.cov_kernel, budget)
        # find MI of the greedy solution
        V = set(range(self.cov_kernel.shape[0]))
        A_greedy = set(greedy_sensors)
        V_minus_A_greedy = V - A_greedy
        I_greedy = _compute_mi(self.cov_kernel, A_greedy, V_minus_A_greedy)
        print(f'Greedy solution is {greedy_sensors}, mi is {I_greedy}')

        # online greedy
        eps = 0.1
        num_items = self.cov_kernel.shape[0]
        seqopt = SeqOpt(num_items, eps)
        num_rounds = 100

        for i in tqdm(range(num_rounds)):

            # sample a list of activations from seqopt
            sel_list = seqopt.sample_sequence(with_replacement=False)

            # NOTE: we are going to use the batch covariance
            # every round as this is a toy setting and we want to
            # verify that seqopt is converging to good solutions

            # update seqopt
            compute_marginal_gain_func = create_submod_f(self.cov_kernel)
            seqopt.update(sel_list, compute_marginal_gain_func)

        # now sample a list of ops and hope it is diverse
        seqopt_sensors = seqopt.sample_sequence(with_replacement=False)
        seqopt_sensors = seqopt_sensors[:budget]

        V = set(range(self.cov_kernel.shape[0]))
        A_seqopt = set(seqopt_sensors)
        V_minus_A_seqopt = V - A_seqopt
        I_seqopt = _compute_mi(self.cov_kernel, A_seqopt, V_minus_A_seqopt)
        print(f'SeqOpt solution is {seqopt_sensors}, mi is {I_seqopt}')

        self.assertAlmostEqual(I_seqopt, I_greedy, delta=0.1)
        self.assertAlmostEqual(I_greedy, bf_val, delta=0.1)



def main():
    unittest.main()


    # # generate some synthetic 1d data
    # Y = synthetic_data2()
    # vals = [item[0] for item in Y]
    # print(f'{np.unique(vals).shape[0]} unique observations' )
    # plt.figure()
    # plt.plot(vals)
    # # plt.show()

    # # budget on sensor
    # budget = 4

    # # compute kernel covariance of observations
    # cov_kernel = compute_synthetic_data_covariance(Y)
    # print(f'Det of cov_kernel is {np.linalg.det(cov_kernel)}')

    # plt.figure()
    # sns.heatmap(cov_kernel, annot=False, cmap='coolwarm')
    # # plt.show()

    # # brute force solution
    # bf_sensors, bf_val = compute_brute_force_sol(cov_kernel, budget)
    # print(f'Brute force max subset {bf_sensors}, max mi {bf_val}')

    # # greedy
    # greedy_sensors = aa.greedy_op_selection(cov_kernel, budget)
    # # find MI of the greedy solution
    # V = set(range(cov_kernel.shape[0]))
    # A_greedy = set(greedy_sensors)
    # V_minus_A_greedy = V - A_greedy
    # I_greedy = _compute_mi(cov_kernel, A_greedy, V_minus_A_greedy)
    # print(f'Greedy solution is {greedy_sensors}, mi is {I_greedy}')

    # # online greedy
    # eps = 0.1
    # num_items = cov_kernel.shape[0]
    # seqopt = SeqOpt(num_items, eps)
    # num_rounds = 100
    # for i in range(num_rounds):
    #     print(f'Round {i}/{num_rounds}')

    #     # sample a list of activations from seqopt
    #     sel_list = seqopt.sample_sequence(with_replacement=False)

    #     # NOTE: we are going to use the batch covariance
    #     # every round as this is a toy setting and we want to
    #     # verify that seqopt is converging to good solutions

    #     # update seqopt
    #     compute_marginal_gain_func = create_submod_f(cov_kernel)
    #     seqopt.update(sel_list, compute_marginal_gain_func)

    # # now sample a list of ops and hope it is diverse
    # seqopt_sensors = seqopt.sample_sequence(with_replacement=False)
    # V = set(range(cov_kernel.shape[0]))
    # A_seqopt = set(seqopt_sensors)
    # V_minus_A_seqopt = V - A_seqopt
    # I_seqopt = _compute_mi(cov_kernel, A_seqopt, V_minus_A_seqopt)
    # print(f'SeqOpt solution is {seqopt_sensors}, mi is {I_seqopt}')



if __name__ == '__main__':
    main()



