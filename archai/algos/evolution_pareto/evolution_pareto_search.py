from abc import ABCMeta, abstractmethod
from overrides.overrides import overrides

import random
from typing import Tuple, List

import torch.nn as nn

from archai.common.common import logger
from archai.nas.arch_meta import ArchWithMetaData
from archai.nas.discrete_search_space import DiscreteSearchSpace
from archai.nas.searcher import Searcher, SearchResult
from archai.common.config import Config


class EvolutionParetoSearch(Searcher):


    @abstractmethod
    def get_search_space(self)->DiscreteSearchSpace:
        pass


    @abstractmethod
    def calc_memory_latency(self, population:List[ArchWithMetaData])->None:
        # computes memory and latency of each model
        # and updates the meta data
        pass

    @abstractmethod
    def calc_task_accuracy(self, population:List[ArchWithMetaData])->None:
        # computes task accuracy of each model
        # and updates the meta data
        pass


    @abstractmethod
    def update_pareto_frontier(self, population:List[ArchWithMetaData])->List[ArchWithMetaData]:
        pass


    @abstractmethod
    def mutate_parents(self, parents:List[ArchWithMetaData])->List[ArchWithMetaData]:
        pass


    @abstractmethod
    def crossover_parents(self, parents:List[ArchWithMetaData])->List[ArchWithMetaData]:
        pass


    @abstractmethod
    def plot_search_state(self, all_pop:List[ArchWithMetaData], pareto:List[ArchWithMetaData], iter_num:int)->None:
        pass

    @abstractmethod
    def save_search_status(self, all_pop:List[ArchWithMetaData], pareto:List[ArchWithMetaData], iter_num:int)->None:
        pass


    def _sample_init_population(self)->List[ArchWithMetaData]:
        init_pop:List[ArchWithMetaData] = []
        while len(init_pop) < self.init_num_models:
            init_pop.append(self.search_space.random_sample())  
        return init_pop


    def _sample_random_to_mix(self)->List[ArchWithMetaData]:
        mix_pop:List[ArchWithMetaData] = []
        while len(mix_pop) < self.num_random_mix:
            mix_pop.append(self.search_space.random_sample())
        return mix_pop

    def on_calc_task_accuracy_end(self, current_pop: List[ArchWithMetaData]) -> None:
        ''' Callback function called right after calc_task_accuracy()'''
        pass

    @overrides
    def search(self, conf_search:Config):
        
        self.init_num_models = conf_search['init_num_models']
        self.num_iters = conf_search['num_iters']
        self.num_random_mix = conf_search['num_random_mix']
        self.max_unseen_population = conf_search['max_unseen_population']
        
        assert self.init_num_models > 0 
        assert self.num_iters > 0
        assert self.num_random_mix > 0
        assert self.max_unseen_population > 0

        self.search_space = self.get_search_space()
        assert isinstance(self.search_space, DiscreteSearchSpace)

        # sample the initial population
        unseen_pop:List[ArchWithMetaData] = self._sample_init_population()

        self.iter_num = -1

        self.all_pop = unseen_pop
        for i in range(self.num_iters):
            self.iter_num = i
            logger.info(f'starting evolution pareto iter {i}')
            
            # for the unseen population 
            # calculates the memory and latency
            # and inserts it into the meta data of each member
            logger.info(f'iter {i}: calculating memory latency for {len(unseen_pop)} models') 
            self.calc_memory_latency(unseen_pop)

            # calculate task accuracy proxy
            # could be anything from zero-cost proxy
            # to partial training
            logger.info(f'iter {i}: calculating task accuracy for {len(unseen_pop)} models')
            self.calc_task_accuracy(unseen_pop)  
            self.on_calc_task_accuracy_end(unseen_pop)

            # update the pareto frontier
            logger.info(f'iter {i}: updating the pareto')
            pareto:List[ArchWithMetaData] = self.update_pareto_frontier(self.all_pop)
            logger.info(f'iter {i}: found {len(pareto)} members')

            # select parents for the next iteration from 
            # the current estimate of the frontier while
            # giving more weight to newer parents
            # TODO
            parents = pareto # for now
            logger.info(f'iter {i}: chose {len(parents)} parents')

            # plot the state of search
            self.save_search_status(all_pop=self.all_pop, pareto=pareto, iter_num=i)
            self.plot_search_state(all_pop=self.all_pop, pareto=pareto, iter_num=i)

            # mutate random 'k' subsets of the parents
            # while ensuring the mutations fall within 
            # desired constraint limits
            mutated = self.mutate_parents(parents)
            logger.info(f'iter {i}: mutation yielded {len(mutated)} new models')

            # crossover random 'k' subsets of the parents
            # while ensuring the mutations fall within 
            # desired constraint limits
            crossovered = self.crossover_parents(parents)
            logger.info(f'iter {i}: crossover yielded {len(crossovered)} new models')

            # sample some random samples to add to the parent mix 
            # to mitigage local minima
            rand_mix = self._sample_random_to_mix()

            unseen_pop = crossovered + mutated + rand_mix
            # shuffle before we pick a smaller population for the next stage
            random.shuffle(unseen_pop)
            logger.info(f'iter {i}: total unseen population before restriction {len(unseen_pop)}')
            unseen_pop = unseen_pop[:self.max_unseen_population]
            logger.info(f'iter {i}: total unseen population after restriction {len(unseen_pop)}')

            # update the set of architectures ever visited
            self.all_pop.extend(unseen_pop)

            
            



            





    