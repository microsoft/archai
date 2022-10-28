# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from overrides import overrides
from typing import List, Dict, Union, Optional
from pathlib import Path

import numpy as np

from archai.common.utils import create_logger
from archai.discrete_search import (
    ArchaiModel,  Objective, AsyncObjective,
    SearchResults, get_non_dominated_sorting, evaluate_models
)
from archai.discrete_search import BayesOptSearchSpace, EvolutionarySearchSpace
from archai.discrete_search.api.predictor import Predictor, MeanVar
from archai.discrete_search.predictors import PredictiveDNNEnsemble
from archai.discrete_search.api.dataset import DatasetProvider
from archai.discrete_search.api.searcher import Searcher


class MoBananasSearch(Searcher):
    def __init__(self, output_dir: str,
                 search_space: BayesOptSearchSpace, 
                 objectives: Dict[str, Union[Objective, AsyncObjective]], 
                 dataset_provider: DatasetProvider,
                 surrogate_model: Optional[Predictor] = None,
                 cheap_objectives: Optional[List[str]] = None,
                 num_iters: int = 10, init_num_models: int = 10,
                 num_parents: int = 10, mutations_per_parent: int = 5,
                 num_mutations: int = 10, seed: int = 1):

        assert isinstance(search_space, BayesOptSearchSpace)
        assert isinstance(search_space, EvolutionarySearchSpace)
        
        if surrogate_model:
            assert isinstance(surrogate_model, Predictor)
        else:
            surrogate_model = PredictiveDNNEnsemble()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.search_space = search_space
        self.dataset_provider = dataset_provider
        self.surrogate_model = surrogate_model

        # Objectives
        self.objectives = objectives
        self.cheap_objectives = cheap_objectives or []
        self.expensive_objectives = sorted([
            obj for obj in self.objectives if obj not in self.cheap_objectives
        ])

        # Algorithm parameters
        self.num_iters = num_iters
        self.init_num_models = init_num_models
        self.num_parents = num_parents
        self.mutations_per_parent = mutations_per_parent
        self.num_mutations = num_mutations

        # Utils
        self.logger = create_logger(str(self.output_dir / 'log.log'), enable_stdout=True) 
        self.evaluated_archids = set()
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.surrogate_dataset = []
        self.search_state = SearchResults(search_space, objectives)

        
    def calc_cheap_objectives(self, archs: List[ArchaiModel]) -> Dict[str, np.ndarray]:
        cheap_objectives = {
            obj_name: obj 
            for obj_name, obj in self.objectives.items()
            if obj_name in self.cheap_objectives
        }

        return evaluate_models(archs, cheap_objectives, self.dataset_provider)

    def get_surrogate_iter_dataset(self, all_pop: List[ArchaiModel]):
        encoded_archs = np.vstack([self.search_space.encode(m) for m in all_pop])
        target = np.array([
            self.search_state.all_evaluation_results[obj] 
            for obj in self.expensive_objectives
        ]).T

        return encoded_archs, target
    
    def mutate_parents(self, parents: List[ArchaiModel],
                       mutations_per_parent: int = 1) -> List[ArchaiModel]:
        mutated_models = [
            self.search_space.mutate(p)
            for p in parents
            for _ in range(mutations_per_parent)
        ]

        # Removes duplicates
        mutated_models = [
            m for m in mutated_models 
            if m.archid not in self.evaluated_archids
        ]

        if not mutated_models:
            raise ValueError(
                'Mutations yielded 0 new models. '
                'Try increasing `num_parents` and `mutations_per_parent` parameters'
            )

        return mutated_models

    def predict_expensive_objectives(self, archs: List[ArchaiModel]) -> Dict[str, MeanVar]:
        ''' Predicts expensive objectives for `archs` using surrogate model ''' 
        encoded_archs = np.vstack([self.search_space.encode(m) for m in archs])
        pred_results = self.surrogate_model.predict(encoded_archs)
        
        return {
            obj_name: MeanVar(pred_results.mean[:, i], pred_results.var[:, i])
            for i, obj_name  in enumerate(self.expensive_objectives)
        }

    def thompson_sampling(self, archs: List[ArchaiModel], sample_size: int,
                          pred_expensive_objs: Dict[str, MeanVar],
                          cheap_objs: Dict[str, np.ndarray]) -> List[int]:
        ''' Returns the selected architecture list indices from Thompson Sampling  '''                           
        simulation_results = cheap_objs

        # Simulates results from surrogate model assuming N(pred_mean, pred_std)
        simulation_results.update({
            obj_name: self.rng.randn(len(archs)) * np.sqrt(pred.var) + pred.mean
            for obj_name, pred in pred_expensive_objs.items()
        })

        # Performs non-dominated sorting
        # TODO: Shuffle elements inside each frontier to avoid giving advantage to a specific part of the pareto
        # or add crowd-sorting
        nds_frontiers = get_non_dominated_sorting(archs, simulation_results, self.objectives)
        
        return [
            idx for frontier in nds_frontiers
            for idx in frontier['indices']
        ][:sample_size]

    @overrides
    def search(self):
        all_pop, selected_indices, pred_expensive_objs = [], [], {}
        unseen_pop = [self.search_space.random_sample() for _ in range(self.init_num_models)]

        for i in range(self.num_iters):
            self.logger.info(f'Starting iteration {i}')
            all_pop.extend(unseen_pop)

            self.logger.info(f'Evaluating objectives for {len(unseen_pop)} architectures')
            iter_results = evaluate_models(unseen_pop, self.objectives, self.dataset_provider)

            self.evaluated_archids.update([m.archid for m in unseen_pop])
            
            # Adds iteration results and predictions from the previous iteration for comparison
            extra_model_data = {
                f'Predicted {obj_name} {c}': getattr(obj_results, c)[selected_indices]
                for obj_name, obj_results in pred_expensive_objs.items()
                for c in ['mean', 'var']
            }
            self.search_state.add_iteration_results(unseen_pop, iter_results, extra_model_data)

            # Updates surrogate
            self.logger.info('Updating surrogate model...')
            X, y = self.get_surrogate_iter_dataset(all_pop)
            self.surrogate_model.fit(X, y)

            # Selects top-`num_parents` models from non-dominated sorted results
            nds_frontiers = get_non_dominated_sorting(
                all_pop,
                self.search_state.all_evaluation_results, 
                self.objectives
            )
            parents = [model for frontier in nds_frontiers for model in frontier['models']]
            parents = parents[:self.num_parents]

            # Mutates top models
            self.logger.info(f'Generating mutations for {len(parents)} parent architectures...')
            mutated = self.mutate_parents(parents, self.mutations_per_parent)

            # Predicts expensive objectives using surrogate model 
            # and calculates cheap objectives for mutated architectures
            self.logger.info(f'Predicting objectives {str(self.expensive_objectives)} using surrogate model')
            pred_expensive_objs = self.predict_expensive_objectives(mutated)

            self.logger.info(f'Calculating cheap objectives {str(self.cheap_objectives)}')
            cheap_objs = self.calc_cheap_objectives(mutated)

            # Selects `num_mutations`-archtiectures for next iteration using Thompson Sampling
            selected_indices = self.thompson_sampling(
                mutated, self.num_mutations,
                pred_expensive_objs, cheap_objs
            )
            unseen_pop = [mutated[i] for i in selected_indices]
            
            self.logger.info(f'{self.num_mutations} candidate architectures were selected for the next iteration')

            # Save plots and reports
            self.search_state.save_all_2d_pareto_evolution_plots(self.output_dir)
            self.search_state.save_search_state(self.output_dir / f'search_state_{i}.csv')

        return self.search_state
