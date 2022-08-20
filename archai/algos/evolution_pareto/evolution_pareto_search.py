from abc import ABCMeta, abstractmethod
from overrides.overrides import overrides
from pathlib import Path

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
    def calc_secondary_objectives(self, population:List[ArchWithMetaData])->None:
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
    def mutate_parents(self, parents:List[ArchWithMetaData], mutations_per_parent: int = 1)->List[ArchWithMetaData]:
        pass


    @abstractmethod
    def crossover_parents(self, parents:List[ArchWithMetaData], num_crossovers: int = 1)->List[ArchWithMetaData]:
        pass


    @abstractmethod
    def plot_search_state(self, all_pop:List[ArchWithMetaData], pareto:List[ArchWithMetaData], iter_num:int)->None:
        pass

    @abstractmethod
    def save_search_status(self, all_pop:List[ArchWithMetaData], pareto:List[ArchWithMetaData], iter_num:int)->None:
        pass

    def read_search_status(self, search_state:Path)->Tuple[List[ArchWithMetaData], List[ArchWithMetaData], int]:
        ''' Optional method for loading the search status from a previous search. '''
        raise NotImplementedError

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

    def on_search_iteration_start(self, current_pop: List[ArchWithMetaData]) -> None:
        ''' Callback function called right before each search iteration'''
        pass
    
    def on_search_end(self, pareto_models: List[ArchWithMetaData]) -> None:
        ''' Callback function called after the search has finished '''
        pass

    def select_next_population(self, current_pop: List[ArchWithMetaData]) -> List[ArchWithMetaData]:
        random.shuffle(current_pop)
        return current_pop[:self.max_unseen_population]

    @overrides
    def search(self, conf_search:Config):
        self.init_num_models = conf_search['init_num_models']
        self.num_iters = conf_search['num_iters']
        self.num_random_mix = conf_search['num_random_mix']
        self.max_unseen_population = conf_search['max_unseen_population']
        self.mutations_per_parent = conf_search.get('mutations_per_parent', 1)
        self.num_crossovers = conf_search.get('num_crossovers', 1)

        assert self.init_num_models > 0 
        assert self.num_iters > 0
        assert self.num_random_mix > 0
        assert self.max_unseen_population > 0

        self.search_space = self.get_search_space()
        assert isinstance(self.search_space, DiscreteSearchSpace)
        
        # Gets the previous search state path from search config parameter `resume_search_from`.
        # If previous search state path is passed in, make sure read_search_status method is implemented.
        prev_search_state = conf_search.get('resume_search_from', None)

        if prev_search_state:
            # Load population from previously saved search state and continue search
            seen_pop, pareto, last_iter = self.read_search_status(Path(prev_search_state))
            self.all_pop = seen_pop
            curr_iter = last_iter + 1
            unseen_pop = self._populate_next_generation(pareto, curr_iter)
        else:
            # sample the initial population
            self.iter_num = 0
            curr_iter = 0
            unseen_pop:List[ArchWithMetaData] = self._sample_init_population()
            self.all_pop = unseen_pop

        while curr_iter < self.num_iters:
            self.iter_num = curr_iter + 1

            logger.info(f'starting evolution pareto iter {curr_iter}')
            self.on_search_iteration_start(unseen_pop)
            
            # for the unseen population 
            # calculates the memory and latency
            # and inserts it into the meta data of each member
            logger.info(f'iter {curr_iter}: calculating memory latency for {len(unseen_pop)} models') 
            self.calc_secondary_objectives(unseen_pop)

            # calculate task accuracy proxy
            # could be anything from zero-cost proxy
            # to partial training
            logger.info(f'iter {curr_iter}: calculating task accuracy for {len(unseen_pop)} models')
            self.calc_task_accuracy(unseen_pop)  
            self.on_calc_task_accuracy_end(unseen_pop)

            # update the pareto frontier
            logger.info(f'iter {curr_iter}: updating the pareto')
            pareto:List[ArchWithMetaData] = self.update_pareto_frontier(self.all_pop)
            logger.info(f'iter {curr_iter}: found {len(pareto)} members')

            # plot the state of search
            self.save_search_status(all_pop=self.all_pop, pareto=pareto, iter_num=curr_iter)
            self.plot_search_state(all_pop=self.all_pop, pareto=pareto, iter_num=curr_iter)

            unseen_pop = self._populate_next_generation(pareto, curr_iter)

            curr_iter += 1
        
        self.on_search_end(pareto)

    def _populate_next_generation(self, pareto: List[ArchWithMetaData], iter: int)->List[ArchWithMetaData]:
        # select parents for the next iteration from 
        # the current estimate of the frontier while
        # giving more weight to newer parents
        # TODO
        parents = pareto # for now
        logger.info(f'iter {iter}: chose {len(parents)} parents')

        # mutate random 'k' subsets of the parents
        # while ensuring the mutations fall within 
        # desired constraint limits
        mutated = self.mutate_parents(parents, self.mutations_per_parent)
        logger.info(f'iter {iter}: mutation yielded {len(mutated)} new models')

        # crossover random 'k' subsets of the parents
        # while ensuring the mutations fall within 
        # desired constraint limits
        crossovered = self.crossover_parents(parents, self.num_crossovers)
        logger.info(f'iter {iter}: crossover yielded {len(crossovered)} new models')

        # sample some random samples to add to the parent mix 
        # to mitigage local minima
        rand_mix = self._sample_random_to_mix()

        unseen_pop = crossovered + mutated + rand_mix
        # shuffle before we pick a smaller population for the next stage
        logger.info(f'iter {iter}: total unseen population before restriction {len(unseen_pop)}')
        unseen_pop = self.select_next_population(unseen_pop)
        logger.info(f'iter {iter}: total unseen population after restriction {len(unseen_pop)}')

        # update the set of architectures ever visited
        self.all_pop.extend(unseen_pop)

        return unseen_pop

    def finalize (self, final_iter, arch_ids : List) -> None:

        self.search_space = self.get_search_space()
        assert isinstance(self.search_space, DiscreteSearchSpace)
        unseen_pop:List[ArchWithMetaData] = [self.search_space._sample_archid (curr_archid) for curr_archid in arch_ids]

        self.all_pop = unseen_pop
        self.on_search_iteration_start(unseen_pop)
        final_iter = final_iter + 1
            
        logger.info(f'iter {final_iter}: calculating memory latency for {len(unseen_pop)} models') 
        self.calc_secondary_objectives(unseen_pop)

        # calculate task accuracy proxy
        # could be anything from zero-cost proxy
        # to partial training
        logger.info(f'iter {final_iter}: calculating task accuracy for {len(unseen_pop)} models')
        self.calc_task_accuracy(unseen_pop)  
        logger.info(f'self.all_pop len = {len(self.all_pop)}; 3')
        self.on_calc_task_accuracy_end(unseen_pop)

        # update the pareto frontier
        logger.info(f'iter {final_iter}: updating the pareto')
        logger.info(f'self.all_pop len = {len(self.all_pop)}; 4')
        pareto:List[ArchWithMetaData] = self.update_pareto_frontier(self.all_pop)
        logger.info(f'iter {final_iter}: found {len(pareto)} members')

            # plot the state of search
        self.save_search_status(all_pop=self.all_pop, pareto=pareto, iter_num=final_iter)
        self.plot_search_state(all_pop=self.all_pop, pareto=pareto, iter_num=final_iter)    