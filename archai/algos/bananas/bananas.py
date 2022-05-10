from abc import ABCMeta, abstractmethod
from overrides.overrides import overrides

import abc
import math as ma
from typing import Tuple, List

import torch.nn as nn

from archai.common.common import logger
from archai.nas.arch_meta import ArchWithMetaData
from archai.nas.discrete_search_space import DiscreteSearchSpace
from archai.nas.searcher import Searcher, SearchResult
from archai.common.config import Config


class Bananas(Searcher):

    @abstractmethod
    def get_search_space(self)->DiscreteSearchSpace:
        pass

    


    @overrides
    def search(self,  conf_search:Config):

        self.init_num_models = conf_search['init_num_models']
        self.num_iters = conf_search['num_iters']

        assert self.init_num_models > 0 
        assert self.num_iters > 0

        self.search_space = self.get_search_space()
        assert isinstance(self.search_space, DiscreteSearchSpace)

        # draw initial sample of architectures 
        unseen_pop:List[ArchWithMetaData] = self._sample_init_population()

        self.all_pop = unseen_pop
        for i in range(self.num_iters):

            self.calc_task_accuracy(unseen_pop)

            self.update_predictive_function()

            # select top 'k' architectures
            # from the seen population
            parents = self.select_top_seen()

            # generate a set of architectures
            # by randomly mutating the top 
            # performing architectures from population so far
            mutated = self.mutate(parents)

            # evaluate the acquisition function 
            # on new population and select 
            # the one with the minimum value for evaluation
            unseen_pop = self.select_best_acquisition(mutated)

            # update the set of architectures ever visited
            self.all_pop.extend(unseen_pop)

