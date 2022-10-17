# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from typing import List, Optional, Set, Dict

from archai.algos.divnas.wmr import Wmr


class SeqOpt:
    """ Implements SeqOpt 
        TODO: Later on we might want to refactor this class 
        to be able to handle bandit feedback """

    def __init__(self, num_items:int, eps:float):
        self._num_items = num_items

        # initialize wmr copies
        self._expert_algos = [Wmr(self._num_items, eps) for i in range(self._num_items)]


    def sample_sequence(self, with_replacement=False)->List[int]:

        sel_set = set()
        # to keep order information
        sel_list = []

        counter = 0
        counter_limit = 10000

        for i in range(self._num_items):
            item_id = self._expert_algos[i].sample()
            if not with_replacement:
                # NOTE: this might be an infinite while loop
                while item_id in sel_set and counter < counter_limit:
                    item_id = self._expert_algos[i].sample()
                    counter += 1
                    
                if counter >= counter_limit:
                    print('Got caught in infinite loop for a while')

            sel_set.add(item_id)
            sel_list.append(item_id)

        return sel_list


    def _check_marg_gains(self, reward_storage:List[List[float]])->bool:
        reward_array = np.array(reward_storage)

        is_descending = True
        for i in range(reward_array.shape[1]):
            marg_gains_this_item = reward_array[:,i]
            is_descending = np.all(np.diff(marg_gains_this_item)<=0)
            if not is_descending:
                return is_descending

        return is_descending


    def _scale_minus_one_to_one(self, rewards:np.array)->np.array:
        scaled = np.interp(rewards, (rewards.min(), rewards.max()), (-1, 1))
        return scaled

    def update(self, sel_list:List[int], compute_marginal_gain_func)->None:
        """ In the full information case we will update 
        all expert copies according to the marginal benefits """

        # mother set
        S = set([i for i in range(self._num_items)])

        reward_storage = []

        # for each slot    
        for slot_id in range(self._num_items):
            # for each action in the slot
            sub_sel = set(sel_list[:slot_id])
            reward_vector = []
            for item in range(self._num_items):                
                # the function passed in 
                # must already be bound to the 
                # covariance function needed
                reward = compute_marginal_gain_func(item, sub_sel, S)
                reward_vector.append(reward)
            
            # update the expert algo copy for this slot
            scaled_rewards = self._scale_minus_one_to_one(np.array(reward_vector))
            self._expert_algos[slot_id].update(scaled_rewards)

            reward_storage.append(reward_vector)

        # # Uncomment to aid in debugging
        # np.set_printoptions(precision=3, suppress=True)
        # print('Marginal gain array (item_id X slots)')
        # print(np.array(reward_storage).T)

        # is_descending = self._check_marg_gains(reward_storage)
        # if not is_descending:
        #     print('WARNING marginal gains are not diminishing')
                
            
        






