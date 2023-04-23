# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from pathlib import Path
from typing import Optional

from overrides import overrides

from archai.api.dataset_provider import DatasetProvider
from archai.common.ordered_dict_logger import OrderedDictLogger
from archai.discrete_search.api.search_objectives import SearchObjectives
from archai.discrete_search.api.search_results import SearchResults
from archai.discrete_search.api.search_space import DiscreteSearchSpace
from archai.discrete_search.api.searcher import Searcher
from archai.discrete_search.utils.multi_objective import get_non_dominated_sorting

logger = OrderedDictLogger(source=__name__)


class SuccessiveHalvingSearch(Searcher):
    """Successive Halving algorithm"""

    def __init__(
        self,
        search_space: DiscreteSearchSpace,
        objectives: SearchObjectives,
        dataset_provider: DatasetProvider,
        output_dir: str,
        num_iters: Optional[int] = 10,
        init_num_models: Optional[int] = 10,
        init_budget: Optional[float] = 1.0,
        budget_multiplier: Optional[float] = 2.0,
        seed: Optional[int] = 1,
    ) -> None:
        """Initialize the Successive Halving.

        Args:
            search_space: Discrete search space.
            search_objectives: Search objectives.
            dataset_provider: Dataset provider.
            output_dir: Output directory.
            num_iters: Number of iterations.
            init_num_models: Number of initial models to evaluate.
            init_budget: Initial budget.
            budget_multiplier: Budget multiplier.
            seed: Random seed.

        """
        super(SuccessiveHalvingSearch, self).__init__()

        assert isinstance(search_space, DiscreteSearchSpace)

        # Search parameters
        self.search_space = search_space
        self.objectives = objectives
        self.dataset_provider = dataset_provider
        self.output_dir = Path(output_dir)
        self.num_iters = num_iters
        self.init_num_models = init_num_models
        self.init_budget = init_budget
        self.budget_multiplier = budget_multiplier

        self.output_dir.mkdir(exist_ok=True)

        # Utils
        self.iter_num = 0
        self.num_sampled_models = 0
        self.seed = seed
        self.search_state = SearchResults(search_space, objectives)
        self.rng = random.Random(seed)

        self.output_dir.mkdir(exist_ok=True, parents=True)

    @overrides
    def search(self) -> SearchResults:
        current_budget = self.init_budget
        population = [self.search_space.random_sample() for _ in range(self.init_num_models)]
        selected_models = population

        for i in range(self.num_iters):
            if len(selected_models) <= 1:
                logger.info(f"Search ended. Architecture selected: {selected_models[0].archid}")
                self.search_space.save_arch(selected_models[0], self.output_dir / "final_model")

                break

            self.on_start_iteration(i + 1)
            logger.info(f"Iteration {i+1}/{self.num_iters}")
            logger.info(f"Evaluating {len(selected_models)} models with budget {current_budget} ...")

            results = self.objectives.eval_all_objs(
                selected_models,
                budgets={obj_name: current_budget for obj_name in self.objectives.objectives},
            )

            # Logs results and saves iteration models
            self.search_state.add_iteration_results(
                selected_models, results, extra_model_data={"budget": [current_budget] * len(selected_models)}
            )

            models_dir = self.output_dir / f"models_iter_{self.iter_num}"
            models_dir.mkdir(exist_ok=True)

            for model in selected_models:
                self.search_space.save_arch(model, str(models_dir / f"{model.archid}"))

            self.search_state.save_search_state(str(self.output_dir / f"search_state_{self.iter_num}.csv"))
            self.search_state.save_all_2d_pareto_evolution_plots(self.output_dir)

            # Keeps only the best `1/self.budget_multiplier` NDS frontiers
            logger.info("Choosing models for the next iteration ...")
            nds_frontiers = get_non_dominated_sorting(selected_models, results, self.objectives)
            nds_frontiers = nds_frontiers[: int(len(nds_frontiers) * 1 / self.budget_multiplier)]

            selected_models = [model for frontier in nds_frontiers for model in frontier["models"]]
            logger.info(f"Kept {len(selected_models)} models for next iteration.")

            # Update parameters for next iteration
            self.iter_num += 1
            current_budget = current_budget * self.budget_multiplier

        return self.search_state
