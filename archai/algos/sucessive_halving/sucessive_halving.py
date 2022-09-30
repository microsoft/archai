import random
from abc import ABCMeta, abstractmethod
from overrides.overrides import overrides
from typing import Tuple, List, Union, Dict, Optional
from tqdm import tqdm
from pathlib import Path

import numpy as np

from archai.common.common import logger
from archai.nas.arch_meta import ArchWithMetaData
from archai.search_spaces.discrete.base import DiscreteSearchSpaceBase
from archai.metrics.base import BaseMetric, BaseAsyncMetric
from archai.metrics import evaluate_models, SearchResults
from archai.nas.searcher import Searcher
from archai.common.config import Config
from archai.datasets.data import DatasetProvider

class SucessiveHalvingAlgo(Searcher):
    def __init__(self, search_space: DiscreteSearchSpaceBase, 
                 objectives: Dict[str, Union[BaseMetric, BaseAsyncMetric]], 
                 dataset_provider: DatasetProvider,
                 output_dir: str, num_iters: int = 10,
                 init_num_models: int = 10,
                 init_budget: float = 1.0,
                 budget_multiplier: float = 2.0,
                 seed: int = 1):
        
        assert isinstance(search_space, DiscreteSearchSpaceBase)
        if len(objectives) > 1:
            raise NotImplementedError(
                'Currently only single objective search is supported'
            )

        # Search parameters
        self.search_space = search_space
        self.objectives = objectives
        self.dataset_provider = dataset_provider
        self.output_dir = Path(output_dir)
        self.num_iters = num_iters
        self.init_num_models = init_num_models
        self.init_budget = init_budget
        self.budget_multiplier = budget_multiplier

        # Utils
        self.iter_num = 0
        self.num_sampled_models = 0
        self.seed = seed
        self.search_state = SearchResults(search_space, objectives)
        self.rng = random.Random(seed)

        self.output_dir.mkdir(exist_ok=True, parents=True)

    def sample_init_models(self, sample_size: int) -> List[ArchWithMetaData]:
        architectures = [
            self.search_space.get([self.seed + i]) for i in range(sample_size)
        ]
        
        self.num_sampled_models += sample_size
        return architectures

    @overrides
    def search(self) -> SearchResults:
        current_budget = self.init_budget
        population = self.sample_init_models(self.init_num_models)
        selected_models = np.ones(len(population)).astype(np.bool8)

        # Only single-objective optimization is currently supported
        objective = list(self.objectives.values())[0]

        for i in range(self.num_iters):

            # Checks if there's only one model left
            if selected_models.sum() <= 1:
                final_model = [model for i, model in enumerate(population) if selected_models[i]][0]
                logger.info(f'Search ended. Architecture selected: {final_model.metadata["archid"]}')
                self.search_space.save_arch(final_model, self.output_dir / 'final_model')
                
                break

            logger.info(
                f'Starting iteration {i} with {selected_models.sum()} architectures '
                f'and budget of {current_budget}.'
            )

            # Doubles budget 
            budgets = {
                obj_name: current_budget
                for obj_name in self.objectives
            }

            # Evaluates objectives
            iter_models = [model for i, model in enumerate(population) if selected_models[i]]
            results = evaluate_models(iter_models, self.objectives, self.dataset_provider, budgets)

            # Removes the bottom `1 - 1/self.budget_multiplier` worst models
            result_arr = list(results.values())[0] 

            if objective.higher_is_better:
                selected_models[selected_models] *= (result_arr >= np.quantile(result_arr, 1 - 1/self.budget_multiplier))
            else:
                selected_models[selected_models] *= (result_arr <= np.quantile(result_arr, 1/self.budget_multiplier))

            logger.info(f'Removing worst {100/self.budget_multiplier:.2f}% model for next iteration...')

            # Logs results
            self.search_state.add_iteration_results(
                iter_models, results,
                extra_model_data={
                    'budget': [current_budget] * len(iter_models)
                }
            )

            # Saves iter_models 
            models_dir = self.output_dir / f'models_iter_{self.iter_num}'
            models_dir.mkdir(exist_ok=True)

            for model in iter_models:
                self.search_space.save_arch(model, str(models_dir / f'{model.metadata["archid"]}'))

            # Saves search state
            self.search_state.save_search_state(
                str(self.output_dir / f'search_state_{self.iter_num}.csv')
            )

            # Update parameters for next iteration
            self.iter_num += 1
            current_budget = current_budget * self.budget_multiplier
        
        return self.search_state

