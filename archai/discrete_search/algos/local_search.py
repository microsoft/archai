# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from pathlib import Path
from typing import List, Optional

from overrides import overrides
from tqdm import tqdm

from archai.api.archai_model import ArchaiModel
from archai.api.dataset_provider import DatasetProvider
from archai.api.search_objectives import SearchObjectives
from archai.api.searcher import Searcher
from archai.common.ordered_dict_logger import OrderedDictLogger
from archai.discrete_search.api.search_results import DiscreteSearchResults
from archai.discrete_search.api.search_space import EvolutionarySearchSpace

logger = OrderedDictLogger(source=__name__)


class LocalSearch(Searcher):
    def __init__(
        self,
        search_space: EvolutionarySearchSpace,
        search_objectives: SearchObjectives,
        dataset_provider: DatasetProvider,
        output_dir: str,
        num_iters: int = 10,
        init_num_models: int = 10,
        initial_population_paths: Optional[List[str]] = None,
        mutations_per_parent: int = 1,
        seed: int = 1,
    ):
        """Local search algorithm. In each iteration, the algorithm generates a new population by
        mutating the current Pareto frontier. The process is repeated until `num_iters` is reached.

        Args:
            search_space (EvolutionarySearchSpace): Discrete search space compatible with evolutionary algorithms
            search_objectives (SearchObjectives): Search objectives
            dataset_provider (DatasetProvider): Dataset provider used to evaluate models
            output_dir (str): Output directory
            num_iters (int, optional): Number of search iterations. Defaults to 10.
            init_num_models (int, optional): Number of initial models. Defaults to 10.
            initial_population_paths (Optional[List[str]], optional): Paths to initial population.
                If None, then `init_num_models` random models are used. Defaults to None.
            mutations_per_parent (int, optional): Number of mutations per parent. Defaults to 1.
            seed (int, optional): Random seed. Defaults to 1.
        """
        assert isinstance(
            search_space, EvolutionarySearchSpace
        ), f"{str(search_space.__class__)} is not compatible with {str(self.__class__)}"

        self.iter_num = 0
        self.search_space = search_space
        self.so = search_objectives
        self.dataset_provider = dataset_provider
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Algorithm settings
        self.num_iters = num_iters
        self.init_num_models = init_num_models
        self.initial_population_paths = initial_population_paths
        self.mutations_per_parent = mutations_per_parent

        # Utils
        self.search_state = DiscreteSearchResults(search_space, self.so)
        self.seed = seed
        self.rng = random.Random(seed)
        self.seen_archs = set()
        self.num_sampled_archs = 0

        assert self.init_num_models > 0
        assert self.num_iters > 0

    def mutate_parents(
        self, parents: List[ArchaiModel], mutations_per_parent: int = 1, patience: int = 20
    ) -> List[ArchaiModel]:
        mutations = {}

        for p in tqdm(parents, desc="Mutating parents"):
            candidates = {}
            nb_tries = 0

            while len(candidates) < mutations_per_parent and nb_tries < patience:
                mutated_model = self.search_space.mutate(p)
                mutated_model.metadata["parent"] = p.archid

                if not self.so.is_model_valid(mutated_model, self.dataset_provider):
                    continue

                if mutated_model.archid not in self.seen_archs:
                    mutated_model.metadata["generation"] = self.iter_num
                    candidates[mutated_model.archid] = mutated_model
                nb_tries += 1
            mutations.update(candidates)

        return list(mutations.values())

    def sample_models(self, num_models: int, patience: int = 5) -> List[ArchaiModel]:
        nb_tries, valid_sample = 0, []

        while len(valid_sample) < num_models and nb_tries < patience:
            sample = [self.search_space.random_sample() for _ in range(num_models)]

            _, valid_indices = self.so.validate_constraints(sample, self.dataset_provider)
            valid_sample += [sample[i] for i in valid_indices]

        return valid_sample[:num_models]

    @overrides
    def search(self) -> DiscreteSearchResults:
        # sample the initial population
        self.iter_num = 0

        if self.initial_population_paths:
            logger.info(f"Loading initial population from {len(self.initial_population_paths)} architectures")
            unseen_pop = [self.search_space.load_arch(path) for path in self.initial_population_paths]
        else:
            logger.info(f"Using {self.init_num_models} random architectures as the initial population")
            unseen_pop = self.sample_models(self.init_num_models)

        self.all_pop = unseen_pop

        for i in range(self.num_iters):
            self.iter_num = i + 1
            logger.info(f"starting iter {i}")

            if len(unseen_pop) == 0:
                logger.info(f"iter {i}: no models to evaluate, stopping search.")
                break

            # Calculates objectives
            logger.info(
                f"iter {i}: calculating search objectives {list(self.so.objs.keys())} for" f" {len(unseen_pop)} models"
            )

            results = self.so.eval_all_objs(unseen_pop, self.dataset_provider)
            self.search_state.add_iteration_results(
                unseen_pop,
                results,
                # Mutation info
                extra_model_data={
                    "parent": [p.metadata.get("parent", None) for p in unseen_pop],
                },
            )

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

            # mutate random 'k' subsets of the parents
            # while ensuring the mutations fall within
            # desired constraint limits
            unseen_pop = self.mutate_parents(pareto, self.mutations_per_parent)
            logger.info(f"iter {i}: mutation yielded {len(unseen_pop)} new models")

            # update the set of architectures ever visited
            self.all_pop.extend(unseen_pop)

        return self.search_state
