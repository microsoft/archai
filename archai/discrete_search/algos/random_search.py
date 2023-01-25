# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from pathlib import Path
from typing import List, Optional

from overrides import overrides

from archai.api.archai_model import ArchaiModel
from archai.api.dataset_provider import DatasetProvider
from archai.api.search_objectives import SearchObjectives
from archai.api.searcher import Searcher
from archai.common.ordered_dict_logger import OrderedDictLogger
from archai.discrete_search.api.search_results import DiscreteSearchResults
from archai.discrete_search.api.search_space import DiscreteSearchSpace

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
        dataset_provider: DatasetProvider,
        output_dir: str,
        num_iters: Optional[int] = 10,
        samples_per_iter: Optional[int] = 10,
        seed: Optional[int] = 1,
    ):
        """Initialize the random search algorithm.

        Args:
            search_space: Discrete search space.
            search_objectives: Search objectives.
            dataset_provider: Dataset provider.
            output_dir: Output directory.
            num_iters: Number of iterations.
            samples_per_iter: Number of samples per iteration.
            seed: Random seed.

        """

        assert isinstance(
            search_space, DiscreteSearchSpace
        ), f"{str(search_space.__class__)} is not compatible with {str(self.__class__)}"

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
        self.search_state = DiscreteSearchResults(search_space, self.so)
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

            _, valid_indices = self.so.validate_constraints(sample, self.dataset_provider)
            valid_sample += [sample[i] for i in valid_indices if sample[i].archid not in self.seen_archs]

        return valid_sample[:num_models]

    @overrides
    def search(self) -> DiscreteSearchResults:
        for i in range(self.num_iters):
            self.iter_num = i + 1
            logger.info(f"starting iter {i}")

            logger.info(f"Sampling {self.samples_per_iter} random models")
            unseen_pop = self.sample_models(self.samples_per_iter)

            # Calculates objectives
            logger.info(
                f"iter {i}: calculating search objectives {list(self.so.objs.keys())} for" f" {len(unseen_pop)} models"
            )

            results = self.so.eval_all_objs(unseen_pop, self.dataset_provider)
            self.search_state.add_iteration_results(unseen_pop, results)

            # Records evaluated archs to avoid computing the same architecture twice
            self.seen_archs.update([m.archid for m in unseen_pop])

            # update the pareto frontier
            logger.info(f"iter {i}: updating Pareto Frontier")
            pareto = self.search_state.get_pareto_frontier()["models"]
            logger.info(f"iter {i}: found {len(pareto)} members")

            # Saves search iteration results
            self.search_state.save_search_state(str(self.output_dir / f"search_state_{self.iter_num}.csv"))
            self.search_state.save_pareto_frontier_models(str(self.output_dir / f"pareto_models_iter_{self.iter_num}"))
            self.search_state.save_all_2d_pareto_evolution_plots(str(self.output_dir))

        return self.search_state
