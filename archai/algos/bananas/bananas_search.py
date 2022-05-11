from abc import ABCMeta, abstractmethod
from overrides.overrides import overrides

import abc
import math as ma
from typing import Tuple, List

import torch.nn as nn

from archai.common.common import logger
from archai.nas.arch_meta import ArchWithMetaData
from archai.nas.discrete_search_space import DiscreteSearchSpace
from archai.nas.predictive_function import PredictiveFunction
from archai.nas.searcher import Searcher, SearchResult
from archai.common.config import Config


class BananasSearch(Searcher):

    @abstractmethod
    def get_search_space(self)->DiscreteSearchSpace:
        pass


    @abstractmethod
    def get_predictive_obj(self)->PredictiveFunction:
        pass


    @abstractmethod
    def calc_task_accuracy(self, population:List[ArchWithMetaData])->None:
        # computes task accuracy of each model
        # and updates the meta data
        pass


    @abstractmethod
    def update_predictive_function(self)->None:
        # takes the dataset of evaluated architectures
        # and updates the predictive function
        # assumes the dataset is being stored in the class
        pass

    
    @abstractmethod
    def select_top_seen(self)->List[ArchWithMetaData]:
        # amongst all evaluated architectures
        # returns the top performing ones
        pass


    @abstractmethod
    def mutate_parents(self, parents:List[ArchWithMetaData], mutations_per_parent: int = 1)->List[ArchWithMetaData]:
        # mutates each parent architecture
        pass


    @abstractmethod
    def select_best_acquisition(self, mutated:List[ArchWithMetaData])->ArchWithMetaData:
        # returns the architecture that has the best acquisition function
        pass 

    
    def _sample_init_population(self)->List[ArchWithMetaData]:
        init_pop:List[ArchWithMetaData] = []
        while len(init_pop) < self.init_num_models:
            init_pop.append(self.search_space.random_sample())  
        return init_pop



    @overrides
    def search(self,  conf_search:Config):

        self.init_num_models = conf_search['init_num_models']
        self.num_iters = conf_search['num_iters']

        assert self.init_num_models > 0 
        assert self.num_iters > 0

        self.search_space = self.get_search_space()
        assert isinstance(self.search_space, DiscreteSearchSpace)

        self.pred_obj = self.get_predictive_obj()
        assert isinstance(self.pred_obj)

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
            mutated = self.mutate_parents(parents)

            # evaluate the acquisition function 
            # on new population and select 
            # the one with the best value (min/max) 
            # for evaluation
            unseen_pop = self.select_best_acquisition(mutated)

            # update the set of architectures ever visited
            self.all_pop.extend(unseen_pop)

