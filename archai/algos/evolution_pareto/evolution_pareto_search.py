import random
from abc import ABCMeta, abstractmethod
from overrides.overrides import overrides
from typing import Tuple, List, Union, Dict, Optional
from tqdm import tqdm
from pathlib import Path

import numpy as np

from archai.common.utils import create_logger
from archai.nas.arch_meta import ArchWithMetaData
from archai.search_spaces.discrete.base import EvolutionarySearchSpaceBase
from archai.metrics.base import BaseMetric, BaseAsyncMetric
from archai.metrics import evaluate_models, SearchResults
from archai.nas.searcher import Searcher
from archai.common.config import Config
from archai.datasets.data import DatasetProvider


class EvolutionParetoSearch(Searcher):
    def __init__(self, search_space: EvolutionarySearchSpaceBase, 
                 objectives: Dict[str, Union[BaseMetric, BaseAsyncMetric]], 
                 dataset_provider: DatasetProvider,
                 output_dir: str,
                 num_iters: int = 10, init_num_models: int = 10,
                 init_pop_from_paths: Optional[List[str]] = None, 
                 num_random_mix: int = 5, max_unseen_population: int = 100,
                 mutations_per_parent: int = 1, num_crossovers: int = 5, 
                 obj_valid_ranges: Optional[List[Tuple[float, float]]] = None,
                 crowd_sorting: Optional[Dict[str, Union[bool, float]]] = None, seed: int = 1):
        
        assert isinstance(search_space, EvolutionarySearchSpaceBase), \
            f'{str(search_space.__class__)} is not compatible with {str(self.__class__)}'
        
        self.iter_num = 0
        self.search_space = search_space
        self.objectives = objectives
        self.dataset_provider = dataset_provider
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Algorithm settings
        self.num_iters = num_iters
        self.init_num_models = init_num_models
        self.init_pop_from_paths = init_pop_from_paths
        self.num_random_mix = num_random_mix
        self.max_unseen_population = max_unseen_population
        self.mutations_per_parent = mutations_per_parent
        self.num_crossovers = num_crossovers
        self.obj_valid_ranges = obj_valid_ranges
        self.crowd_sorting = crowd_sorting

        # Utils
        self.search_state = SearchResults(search_space, objectives)
        self.seed = seed
        self.rng = random.Random(seed)
        self.evaluated_architectures = set()
        self.num_sampled_archs = 0
        self.logger = create_logger(str(self.output_dir / 'log.log'), enable_stdout=True)

        assert self.init_num_models > 0 
        assert self.num_iters > 0
        assert self.num_random_mix > 0
        assert self.max_unseen_population > 0

    def filter_population(self, population: List[ArchWithMetaData]):
        ''' Filter the population based on the objectives constraints '''
        if not self.obj_valid_ranges:
            return population

        return [
            p for p in population 
            if all(
                self.obj_valid_ranges[obj_idx][0] <= score <= self.obj_valid_ranges[obj_idx][1]
                for obj_idx, score in enumerate(p.metadata['objective'])
            )
        ]

    def mutate_parents(self, parents:List[ArchWithMetaData],
                       mutations_per_parent: int = 1,
                       patience: int = 20) -> List[ArchWithMetaData]:
        mutations = {}
        oversample_factor = 1

        if self.crowd_sorting:
            oversample_factor = (
                self.crowd_sorting['oversampling_factor'] if self.crowd_sorting['mutation']
                else 1
            )

        for p in tqdm(parents, desc='Mutating parents'):
            candidates = {}
            nb_tries = 0

            if len(self.filter_population([p])) == 0:
                self.logger.info(
                    f'Model {p.metadata["archid"]} has latency {p.metadata["latency"]}'
                    f' or memory {p.metadata["memory"]} that is too high. Skipping mutation.'
                )

                continue

            while len(candidates) < (mutations_per_parent * oversample_factor) and nb_tries < patience:
                mutated_model = self.search_space.mutate(p)
                mutated_model.metadata['parent'] = p.metadata['archid']

                mutated_models = [mutated_model] if not isinstance(mutated_model, list) else mutated_model

                for nbr in mutated_models:
                    if nbr.metadata['archid'] not in self.evaluated_architectures:
                        nbr.metadata['generation'] = self.iter_num
                        candidates[nbr.metadata['archid']] = nbr
                nb_tries += 1
            
            # TODO: Figure out a way to use crowd sorting here
            # if candidates and self.crowd_sorting and self.crowd_sorting['mutation']:
            #     candidates_list = list(candidates.items())

            #     secondary_objs_proxy = np.array([
            #         list(self._get_secondary_objectives_proxy(p).values()) for _, p in candidates_list
            #     ])

            #     crowd_dist = compute_crowding_distance(secondary_objs_proxy)
                
            #     # Deletes mutations that are not on the top k
            #     for idx in np.argsort(-crowd_dist, axis=None)[mutations_per_parent:]:
            #         del candidates[candidates_list[idx][0]]

            mutations.update(candidates)

        return list(mutations.values())

    def crossover_parents(self, parents:List[ArchWithMetaData], num_crossovers: int = 1) -> List[ArchWithMetaData]:
        # Randomly samples k distinct pairs from `parents`
        children, children_hashes = [], set()

        if len(parents) >= 2:
            pairs = [random.sample(parents, 2) for _ in range(num_crossovers)]
            for p1, p2 in pairs:
                child = self.search_space.crossover(p1, p2)

                if child:
                    child_id = child.metadata['archid']

                    if child_id not in children_hashes and child_id not in self.evaluated_architectures:
                        child.metadata['generation'] = self.iter_num
                        child.metadata['parents'] = f'{p1.metadata["archid"]},{p2.metadata["archid"]}'
                        children.append(child)
                        children_hashes.add(child_id)

        return children

    def sample_random_models(self, num_models: int) -> List[ArchWithMetaData]:
        mix_pop = []
        
        while len(mix_pop) < num_models:
            self.num_sampled_archs += 1
            mix_pop.append(self.search_space.get([self.seed + self.num_sampled_archs]))

        return mix_pop

    def on_calc_task_accuracy_end(self, current_pop: List[ArchWithMetaData]) -> None:
        ''' Callback function called right after calc_task_accuracy()'''
        pass

    def on_search_iteration_start(self, current_pop: List[ArchWithMetaData]) -> None:
        ''' Callback function called right before each search iteration'''
        pass

    def select_next_population(self, current_pop: List[ArchWithMetaData]) -> List[ArchWithMetaData]:
        random.shuffle(current_pop)
        return current_pop[:self.max_unseen_population]

    @overrides
    def search(self) -> SearchResults:
        # sample the initial population
        self.iter_num = 0
        unseen_pop = self.sample_random_models(self.init_num_models)
        self.all_pop = unseen_pop

        for i in range(self.num_iters):
            self.iter_num = i + 1

            self.logger.info(f'starting evolution pareto iter {i}')
            self.on_search_iteration_start(unseen_pop)

            # Calculates objectives
            self.logger.info(
                f'iter {i}: calculating search objectives {str(self.objectives)} for'
                f' {len(unseen_pop)} models'
            )
            
            results = evaluate_models(unseen_pop, self.objectives, self.dataset_provider)
            self.search_state.add_iteration_results(
                unseen_pop, results,
                
                # Mutation and crossover info
                extra_model_data={
                    'parent': [p.metadata.get('parent', None) for p in unseen_pop],
                    'parents': [p.metadata.get('parents', None) for p in unseen_pop],
                }
            )

            # Records evaluated archs to avoid computing the same architecture twice
            self.evaluated_architectures.update([m.metadata['archid'] for m in unseen_pop])

            # update the pareto frontier
            self.logger.info(f'iter {i}: updating the pareto')
            pareto = self.search_state.get_pareto_frontier()['models']
            self.logger.info(f'iter {i}: found {len(pareto)} members')

            # Saves search iteration results
            self.search_state.save_search_state(
                str(self.output_dir / f'search_state_{self.iter_num}.csv')
            )

            self.search_state.save_pareto_frontier_models(
                str(self.output_dir / f'pareto_models_iter_{self.iter_num}')
            )

            self.search_state.save_all_2d_pareto_evolution_plots(str(self.output_dir))

            # select parents for the next iteration from 
            # the current estimate of the frontier while
            # giving more weight to newer parents
            # TODO
            parents = pareto # for now
            self.logger.info(f'iter {i}: chose {len(parents)} parents')

            # Filters parents
            parents = self.filter_population(parents)
            self.logger.info(f'iter {i}: number of parents after objective fn. filter = {len(parents)}')

            # mutate random 'k' subsets of the parents
            # while ensuring the mutations fall within 
            # desired constraint limits
            mutated = self.mutate_parents(parents, self.mutations_per_parent)
            self.logger.info(f'iter {i}: mutation yielded {len(mutated)} new models')

            # crossover random 'k' subsets of the parents
            # while ensuring the mutations fall within 
            # desired constraint limits
            crossovered = self.crossover_parents(parents, self.num_crossovers)
            self.logger.info(f'iter {i}: crossover yielded {len(crossovered)} new models')

            # sample some random samples to add to the parent mix 
            # to mitigage local minima
            rand_mix = self.sample_random_models(self.num_random_mix)
            unseen_pop = crossovered + mutated + rand_mix

            # shuffle before we pick a smaller population for the next stage
            self.logger.info(f'iter {i}: total unseen population before restriction {len(unseen_pop)}')
            unseen_pop = self.select_next_population(unseen_pop)
            self.logger.info(f'iter {i}: total unseen population after restriction {len(unseen_pop)}')

            # update the set of architectures ever visited
            self.all_pop.extend(unseen_pop)

        return self.search_state
