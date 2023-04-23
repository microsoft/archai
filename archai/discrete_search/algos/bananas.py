# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from overrides import overrides

from archai.api.dataset_provider import DatasetProvider
from archai.common.ordered_dict_logger import OrderedDictLogger
from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.predictor import MeanVar, Predictor
from archai.discrete_search.api.search_objectives import SearchObjectives
from archai.discrete_search.api.search_results import SearchResults
from archai.discrete_search.api.search_space import (
    BayesOptSearchSpace,
    EvolutionarySearchSpace,
)
from archai.discrete_search.api.searcher import Searcher
from archai.discrete_search.predictors.dnn_ensemble import PredictiveDNNEnsemble
from archai.discrete_search.utils.multi_objective import get_non_dominated_sorting

logger = OrderedDictLogger(source=__name__)


class MoBananasSearch(Searcher):
    """Multi-objective version of BANANAS algorithm.

    It has been proposed in `Bag of Baselines for Multi-objective Joint Neural Architecture
    Search and Hyperparameter Optimization`.

    Reference:
        https://arxiv.org/abs/2105.01015

    """

    def __init__(
        self,
        search_space: BayesOptSearchSpace,
        search_objectives: SearchObjectives,
        output_dir: str,
        surrogate_model: Optional[Predictor] = None,
        num_iters: Optional[int] = 10,
        init_num_models: Optional[int] = 10,
        num_parents: Optional[int] = 10,
        mutations_per_parent: Optional[int] = 5,
        num_candidates: Optional[int] = 10,
        clear_evaluated_models: bool = True,
        save_pareto_weights: bool = False,
        seed: Optional[int] = 1,
    ) -> None:
        """Initialize the multi-objective BANANAS.

        Args:
            search_space: Discrete search space compatible with Bayesian Optimization algorithms.
            search_objectives: Search objectives. Expensive objectives (registered with
                `compute_intensive=True`) will be estimated using a surrogate model during
                certain parts of the search. Cheap objectives will be always evaluated directly.
            output_dir: Output directory.
            surrogate_model: Surrogate model. If `None`, a `PredictiveDNNEnsemble` will be used.
            num_iters: Number of iterations.
            init_num_models: Number of initial models to evaluate.
            num_parents: Number of parents to select for each iteration.
            mutations_per_parent: Number of mutations to apply to each parent.
            num_candidates: Number of selected models to add to evaluate in the next iteration.
            clear_evaluated_models: Optimizes memory usage by clearing the architecture
                of `ArchaiModel` after each iteration. Defaults to True
            save_pareto_model_weights: If `False`, saves the weights of the pareto models.
            seed: Random seed.

        """
        super(MoBananasSearch, self).__init__()
        assert isinstance(search_space, BayesOptSearchSpace)
        assert isinstance(search_space, EvolutionarySearchSpace)

        if surrogate_model:
            assert isinstance(surrogate_model, Predictor)
        else:
            surrogate_model = PredictiveDNNEnsemble()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.search_space = search_space
        self.surrogate_model = surrogate_model

        # Objectives
        self.so = search_objectives

        # Algorithm parameters
        self.num_iters = num_iters
        self.init_num_models = init_num_models
        self.num_parents = num_parents
        self.mutations_per_parent = mutations_per_parent
        self.num_candidates = num_candidates

        # Utils
        self.clear_evaluated_models = clear_evaluated_models
        self.save_pareto_weights = save_pareto_weights
        self.seen_archs = set()
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.surrogate_dataset = []
        self.search_state = SearchResults(search_space, search_objectives)

        if self.save_pareto_weights:
            raise NotImplementedError

    def get_surrogate_iter_dataset(self, all_pop: List[ArchaiModel]) -> Tuple[np.ndarray, np.ndarray]:
        """Get the surrogate dataset for the current iteration.

        Args:
            all_pop: All population models.

        Returns:
            Tuple of encoded architectures and target values.

        """

        encoded_archs = np.vstack([self.search_space.encode(m) for m in all_pop])
        target = np.array([self.search_state.all_evaluated_objs[obj] for obj in self.so.expensive_objectives]).T

        return encoded_archs, target

    def sample_models(self, num_models: int, patience: Optional[int] = 30) -> List[ArchaiModel]:
        """Sample models from the search space.

        Args:
            num_models: Number of models to sample.
            patience: Number of tries to sample a valid model.

        Returns:
            List of sampled models.

        """

        nb_tries, valid_sample = 0, []

        while len(valid_sample) < num_models and nb_tries < patience:
            sample = [self.search_space.random_sample() for _ in range(num_models)]

            _, valid_indices = self.so.validate_constraints(sample)
            valid_sample += [sample[i] for i in valid_indices]

        return valid_sample[:num_models]

    def mutate_parents(
        self, parents: List[ArchaiModel], mutations_per_parent: Optional[int] = 1, patience: Optional[int] = 30
    ) -> List[ArchaiModel]:
        """Mutate parents to generate new models.

        Args:
            parents: List of parent models.
            mutations_per_parent: Number of mutations to apply to each parent.
            patience: Number of tries to sample a valid model.

        Returns:
            List of mutated models.

        """

        mutations = {}

        for p in parents:
            candidates = {}
            nb_tries = 0

            while len(candidates) < mutations_per_parent and nb_tries < patience:
                mutated_model = self.search_space.mutate(p)
                mutated_model.metadata["parent"] = p.archid

                if not self.so.is_model_valid(mutated_model):
                    continue

                if mutated_model.archid not in self.seen_archs:
                    candidates[mutated_model.archid] = mutated_model

                nb_tries += 1

            mutations.update(candidates)

        if len(mutations) == 0:
            logger.warn(f"No mutations found after {patience} tries for each one of the {len(parents)} parents.")

        return list(mutations.values())

    def predict_expensive_objectives(self, archs: List[ArchaiModel]) -> Dict[str, MeanVar]:
        """Predict expensive objectives for `archs` using surrogate model.

        Args:
            archs: List of architectures.

        Returns:
            Dictionary of predicted expensive objectives.

        """

        encoded_archs = np.vstack([self.search_space.encode(m) for m in archs])
        pred_results = self.surrogate_model.predict(encoded_archs)

        return {
            obj_name: MeanVar(pred_results.mean[:, i], pred_results.var[:, i])
            for i, obj_name in enumerate(self.so.expensive_objectives)
        }

    def thompson_sampling(
        self,
        archs: List[ArchaiModel],
        sample_size: int,
        pred_expensive_objs: Dict[str, MeanVar],
        cheap_objs: Dict[str, np.ndarray],
    ) -> List[int]:
        """Get the selected architecture list indices from Thompson Sampling.

        Args:
            archs: List of architectures.
            sample_size: Number of architectures to select.
            pred_expensive_objs: Predicted expensive objectives.
            cheap_objs: Cheap objectives.

        Returns:
            List of selected architecture indices.

        """

        simulation_results = cheap_objs

        # Simulates results from surrogate model assuming N(pred_mean, pred_std)
        simulation_results.update(
            {
                obj_name: self.rng.randn(len(archs)) * np.sqrt(pred.var) + pred.mean
                for obj_name, pred in pred_expensive_objs.items()
            }
        )

        # Performs non-dominated sorting
        nds_frontiers = get_non_dominated_sorting(archs, simulation_results, self.so)

        # Shuffle elements inside each frontier to avoid giving advantage to a specific
        # part of the nds frontiers
        for frontier in nds_frontiers:
            self.rng.shuffle(frontier["indices"])

        return [idx for frontier in nds_frontiers for idx in frontier["indices"]][:sample_size]

    @overrides
    def search(self) -> SearchResults:
        all_pop, selected_indices, pred_expensive_objs = [], [], {}
        unseen_pop = self.sample_models(self.init_num_models)

        for i in range(self.num_iters):
            self.on_start_iteration(i + 1)
            logger.info(f"Iteration {i+1}/{self.num_iters}")
            all_pop.extend(unseen_pop)

            logger.info(f"Evaluating objectives for {len(unseen_pop)} architectures ...")
            iter_results = self.so.eval_all_objs(unseen_pop)

            self.seen_archs.update([m.archid for m in unseen_pop])

            # Adds iteration results and predictions from the previous iteration for comparison
            extra_model_data = {
                f"Predicted {obj_name} {c}": getattr(obj_results, c)[selected_indices]
                for obj_name, obj_results in pred_expensive_objs.items()
                for c in ["mean", "var"]
            }
            self.search_state.add_iteration_results(unseen_pop, iter_results, extra_model_data)

            # Clears models from memory if needed
            if self.clear_evaluated_models:
                for m in unseen_pop:
                    m.clear()

            # Updates surrogate
            logger.info("Updating surrogate model ...")
            X, y = self.get_surrogate_iter_dataset(all_pop)
            self.surrogate_model.fit(X, y)

            # Selects top-`num_parents` models from non-dominated sorted results
            nds_frontiers = get_non_dominated_sorting(all_pop, self.search_state.all_evaluated_objs, self.so)
            parents = [model for frontier in nds_frontiers for model in frontier["models"]]
            parents = parents[: self.num_parents]

            # Mutates top models
            logger.info(f"Generating mutations for {len(parents)} parent architectures ...")
            mutated = self.mutate_parents(parents, self.mutations_per_parent)
            logger.info(f"Found {len(mutated)} new architectures satisfying constraints.")

            if not mutated:
                logger.info("No new architectures found. Stopping search ...")
                break

            # Predicts expensive objectives using surrogate model
            # and calculates cheap objectives for mutated architectures
            logger.info(f"Predicting {self.so.expensive_objective_names} for new architectures using surrogate model ...")
            pred_expensive_objs = self.predict_expensive_objectives(mutated)

            logger.info(f"Calculating cheap objectives {self.so.cheap_objective_names} for new architectures ...")
            cheap_objs = self.so.eval_cheap_objs(mutated)

            # Selects `num_candidates`-archtiectures for next iteration using Thompson Sampling
            selected_indices = self.thompson_sampling(mutated, self.num_candidates, pred_expensive_objs, cheap_objs)
            unseen_pop = [mutated[i] for i in selected_indices]

            logger.info(f"Best {self.num_candidates} candidate architectures were selected for the next iteration.")

            # Save plots and reports
            self.search_state.save_all_2d_pareto_evolution_plots(self.output_dir)
            self.search_state.save_search_state(self.output_dir / f"search_state_{i}.csv")

        return self.search_state
