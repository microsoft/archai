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


class EvolutionParetoSearch(Searcher):
    def __init__(
        self,
        search_space: EvolutionarySearchSpace,
        search_objectives: SearchObjectives,
        dataset_provider: DatasetProvider,
        output_dir: str,
        num_iters: int = 10,
        init_num_models: int = 10,
        initial_population_paths: Optional[List[str]] = None,
        num_random_mix: int = 5,
        max_unseen_population: int = 100,
        mutations_per_parent: int = 1,
        num_crossovers: int = 5,
        seed: int = 1,
    ):
        """Evolutionary multi-objective search algorithm that greedily evolves Pareto frontier models
        from a population using mutations and crossovers.

        Starting from an evaluated random subset of models, in each iteration the algorithm evaluates new subset of
        models generated from mutations (`mutation_per_parent`) and crossovers (`num_crossovers`) of the current pareto frontier
        and a new random subset of models (`num_random_mix`). The process is repeated until `num_iters` is reached.

        Args:
            search_space (EvolutionarySearchSpace): Discrete search space compatible with evolutionary algorithms
            search_objectives (SearchObjectives): Search objectives
            dataset_provider (DatasetProvider): Dataset provider
            output_dir (str): Output directory
            num_iters (int, optional): Number of iterations. Defaults to 10.
            init_num_models (int, optional): Number of initial models to evaluate. Defaults to 10.

            initial_population_paths (Optional[List[str]], optional): List of paths to the initial population of
                models. If None, `init_num_models` random models are used. Defaults to None.

            num_random_mix (int, optional): Number of random models to mix with the population
                in each iteration. Defaults to 5.

            max_unseen_population (int, optional): Maximum number of unseen models to evaluate
                in each iteration. Defaults to 100.

            mutations_per_parent (int, optional): Number of distincrt mutations generated for each Pareto
                frontier member. Defaults to 1.

            num_crossovers (int, optional): Total number of crossovers generated per iteration. Defaults to 5.
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
        self.num_random_mix = num_random_mix
        self.max_unseen_population = max_unseen_population
        self.mutations_per_parent = mutations_per_parent
        self.num_crossovers = num_crossovers

        # Utils
        self.search_state = DiscreteSearchResults(search_space, self.so)
        self.seed = seed
        self.rng = random.Random(seed)
        self.seen_archs = set()
        self.num_sampled_archs = 0

        assert self.init_num_models > 0
        assert self.num_iters > 0
        assert self.num_random_mix > 0
        assert self.max_unseen_population > 0

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

    def crossover_parents(
        self, parents: List[ArchaiModel], num_crossovers: int = 1, patience: int = 30
    ) -> List[ArchaiModel]:
        # Randomly samples k distinct pairs from `parents`
        children, children_ids = [], set()

        if len(parents) >= 2:
            pairs = [random.sample(parents, 2) for _ in range(num_crossovers)]
            for p1, p2 in pairs:
                child = self.search_space.crossover([p1, p2])
                nb_tries = 0

                while not self.so.is_model_valid(child, self.dataset_provider) and nb_tries < patience:
                    child = self.search_space.crossover([p1, p2])
                    nb_tries += 1

                if child and self.so.is_model_valid(child, self.dataset_provider):
                    if child.archid not in children_ids and child.archid not in self.seen_archs:
                        child.metadata["generation"] = self.iter_num
                        child.metadata["parents"] = f"{p1.archid},{p2.archid}"
                        children.append(child)
                        children_ids.add(child.archid)

        return children

    def sample_models(self, num_models: int, patience: int = 5) -> List[ArchaiModel]:
        nb_tries, valid_sample = 0, []

        while len(valid_sample) < num_models and nb_tries < patience:
            sample = [self.search_space.random_sample() for _ in range(num_models)]

            _, valid_indices = self.so.validate_constraints(sample, self.dataset_provider)
            valid_sample += [sample[i] for i in valid_indices]

        return valid_sample[:num_models]

    def on_calc_task_accuracy_end(self, current_pop: List[ArchaiModel]) -> None:
        """Callback function called right after calc_task_accuracy()"""

    def on_search_iteration_start(self, current_pop: List[ArchaiModel]) -> None:
        """Callback function called right before each search iteration"""

    def select_next_population(self, current_pop: List[ArchaiModel]) -> List[ArchaiModel]:
        random.shuffle(current_pop)
        return current_pop[: self.max_unseen_population]

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
            self.on_search_iteration_start(unseen_pop)

            # Calculates objectives
            logger.info(
                f"iter {i}: calculating search objectives {list(self.so.objs.keys())} for" f" {len(unseen_pop)} models"
            )

            results = self.so.eval_all_objs(unseen_pop, self.dataset_provider)
            self.search_state.add_iteration_results(
                unseen_pop,
                results,
                # Mutation and crossover info
                extra_model_data={
                    "parent": [p.metadata.get("parent", None) for p in unseen_pop],
                    "parents": [p.metadata.get("parents", None) for p in unseen_pop],
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

            parents = pareto
            logger.info(f"iter {i}: chose {len(parents)} parents")

            # mutate random 'k' subsets of the parents
            # while ensuring the mutations fall within
            # desired constraint limits
            mutated = self.mutate_parents(parents, self.mutations_per_parent)
            logger.info(f"iter {i}: mutation yielded {len(mutated)} new models")

            # crossover random 'k' subsets of the parents
            # while ensuring the mutations fall within
            # desired constraint limits
            crossovered = self.crossover_parents(parents, self.num_crossovers)
            logger.info(f"iter {i}: crossover yielded {len(crossovered)} new models")

            # sample some random samples to add to the parent mix
            # to mitigage local minima
            rand_mix = self.sample_models(self.num_random_mix)
            unseen_pop = crossovered + mutated + rand_mix

            # shuffle before we pick a smaller population for the next stage
            logger.info(f"iter {i}: total unseen population {len(unseen_pop)}")
            unseen_pop = self.select_next_population(unseen_pop)
            logger.info(
                f"iter {i}: total unseen population after `max_unseen_population`" f" restriction {len(unseen_pop)}"
            )

            # update the set of architectures ever visited
            self.all_pop.extend(unseen_pop)

        return self.search_state
