# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from overrides.overrides import overrides
from pathlib import Path
import random
from typing import List, Optional
from tqdm import tqdm

from archai.common.utils import create_logger
from archai.discrete_search import (
    ArchaiModel, DatasetProvider, Searcher, SearchObjectives, SearchResults
)

from archai.discrete_search.api.search_space import DiscreteSearchSpace


class RandomSearch(Searcher):
    def __init__(self, search_space: DiscreteSearchSpace, 
                 search_objectives: SearchObjectives, 
                 dataset_provider: DatasetProvider,
                 output_dir: str, num_iters: int = 10,
                 samples_per_iter: int = 10, seed: int = 1):
        """Random search algorithm that evaluates random samples from the
        search space in each iteration until `num_iters` is reached.

        Args:
            search_space (DiscreteSearchSpace): Discrete search space
            search_objectives (SearchObjectives): Search objectives
            dataset_provider (DatasetProvider): Dataset provider used to evaluate models
            output_dir (str): Output directory
            num_iters (int, optional): Number of search iterations. Defaults to 10.
            samples_per_iter (int, optional): Number of samples per iteration. Defaults to 10.
            seed (int, optional): Random seed. Defaults to 1.
        """        
        assert isinstance(search_space, DiscreteSearchSpace), \
            f'{str(search_space.__class__)} is not compatible with {str(self.__class__)}'
        
        self.iter_num = 0
        self.search_space = search_space
        self.so = search_objectives
        self.dataset_provider = dataset_provider
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Algorithm settings
        self.num_iters = num_iters
        self.samples_per_iter = samples_per_iter

        # Utils
        self.search_state = SearchResults(search_space, self.so)
        self.seed = seed
        self.rng = random.Random(seed)
        self.seen_archs = set()
        self.num_sampled_archs = 0
        self.logger = create_logger(str(self.output_dir / 'log.log'), enable_stdout=True)

        assert self.samples_per_iter > 0 
        assert self.num_iters > 0

    def sample_models(self, num_models: int, patience: int = 5) -> List[ArchaiModel]:
        nb_tries, valid_sample = 0, []

        while len(valid_sample) < num_models and nb_tries < patience:
            sample = [self.search_space.random_sample() for _ in range(num_models)]

            _, valid_indices = self.so.validate_constraints(sample, self.dataset_provider)
            valid_sample += [sample[i] for i in valid_indices 
                             if sample[i].archid not in self.seen_archs]

        return valid_sample[:num_models]

    @overrides
    def search(self) -> SearchResults:
        for i in range(self.num_iters):
            self.iter_num = i + 1
            self.logger.info(f'starting iter {i}')
            
            self.logger.info(f'Sampling {self.samples_per_iter} random models')
            unseen_pop = self.sample_models(self.samples_per_iter)

            # Calculates objectives
            self.logger.info(
                f'iter {i}: calculating search objectives {list(self.so.objs.keys())} for'
                f' {len(unseen_pop)} models'
            )

            results = self.so.eval_all_objs(unseen_pop, self.dataset_provider)
            self.search_state.add_iteration_results(unseen_pop, results)

            # Records evaluated archs to avoid computing the same architecture twice
            self.seen_archs.update([m.archid for m in unseen_pop])

            # update the pareto frontier
            self.logger.info(f'iter {i}: updating Pareto Frontier')
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

        return self.search_state
