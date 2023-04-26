# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from pathlib import Path
from typing import List, Optional

from overrides import overrides

from archai.common.ordered_dict_logger import OrderedDictLogger
from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.search_objectives import SearchObjectives
from archai.discrete_search.api.search_results import SearchResults
from archai.discrete_search.api.search_space import DiscreteSearchSpace
from archai.discrete_search.api.searcher import Searcher

logger = OrderedDictLogger(source=__name__)


class RandomSearch(Searcher):
    """Random search algorithm.

    It evaluates random samples from the search space in each iteration until
    `num_iters` is reached.

    """

    def __init__(
        self,
        search_space: DiscreteSearchSpace,
        search_objectives: SearchObjectives,
        output_dir: str,
        num_iters: Optional[int] = 10,
        samples_per_iter: Optional[int] = 10,
        clear_evaluated_models: Optional[bool] = True,
        save_pareto_model_weights: bool = True,
        seed: Optional[int] = 1,
    ):
        """Initialize the random search algorithm.

        Args:
            search_space: Discrete search space.
            search_objectives: Search objectives.
            output_dir: Output directory.
            num_iters: Number of iterations.
            samples_per_iter: Number of samples per iteration.
            clear_evaluated_models (bool, optional): Optimizes memory usage by clearing the architecture
                of `ArchaiModel` after each iteration. Defaults to True.
            save_pareto_model_weights: If `True`, saves the weights of the pareto models. Defaults to True.
            seed: Random seed.
        """
        super(RandomSearch, self).__init__()
        assert isinstance(
            search_space, DiscreteSearchSpace
        ), f"{str(search_space.__class__)} is not compatible with {str(self.__class__)}"

        self.iter_num = 0
        self.search_space = search_space
        self.so = search_objectives
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Algorithm settings
        self.num_iters = num_iters
        self.samples_per_iter = samples_per_iter

        # Utils
        self.clear_evaluated_models = clear_evaluated_models
        self.save_pareto_model_weights = save_pareto_model_weights
        self.search_state = SearchResults(search_space, self.so)
        self.seed = seed
        self.rng = random.Random(seed)
        self.seen_archs = set()
        self.num_sampled_archs = 0

        assert self.samples_per_iter > 0
        assert self.num_iters > 0

    def sample_models(self, num_models: int, patience: Optional[int] = 5) -> List[ArchaiModel]:
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
            valid_sample += [sample[i] for i in valid_indices if sample[i].archid not in self.seen_archs]

        return valid_sample[:num_models]

    @overrides
    def search(self) -> SearchResults:
        for i in range(self.num_iters):
            self.iter_num = i + 1
            self.on_start_iteration(self.iter_num)
            logger.info(f"Iteration {i+1}/{self.num_iters}")

            logger.info(f"Sampling {self.samples_per_iter} random models ...")
            unseen_pop = self.sample_models(self.samples_per_iter)

            # Calculates objectives
            logger.info(f"Calculating search objectives {list(self.so.objective_names)} for {len(unseen_pop)} models ...")

            results = self.so.eval_all_objs(unseen_pop)
            self.search_state.add_iteration_results(unseen_pop, results)

            # Records evaluated archs to avoid computing the same architecture twice
            self.seen_archs.update([m.archid for m in unseen_pop])

            # update the pareto frontier
            logger.info("Updating Pareto frontier ...")
            pareto = self.search_state.get_pareto_frontier()["models"]
            logger.info(f"Found {len(pareto)} members.")

            # Saves search iteration results
            self.search_state.save_search_state(str(self.output_dir / f"search_state_{self.iter_num}.csv"))
            self.search_state.save_pareto_frontier_models(
                str(self.output_dir / f"pareto_models_iter_{self.iter_num}"),
                save_weights=self.save_pareto_model_weights
            )
            self.search_state.save_all_2d_pareto_evolution_plots(str(self.output_dir))

            # Clears models from memory if needed
            if self.clear_evaluated_models:
                logger.info("Optimzing memory usage ...")
                [model.clear() for model in unseen_pop]

        return self.search_state
