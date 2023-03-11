# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from pathlib import Path
from typing import List, Optional

from overrides import overrides
from tqdm import tqdm

from archai.common.ordered_dict_logger import OrderedDictLogger
from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.search_objectives import SearchObjectives
from archai.discrete_search.api.search_results import SearchResults
from archai.discrete_search.api.search_space import EvolutionarySearchSpace
from archai.discrete_search.api.searcher import Searcher

logger = OrderedDictLogger(source=__name__)


class EvolutionParetoSearch(Searcher):
    """Evolutionary multi-objective search algorithm that greedily
    evolves Pareto frontier models.

    It starts from an evaluated random subset of models. In each iteration, the algorithm
    evaluates new subset of models generated from mutations (`mutation_per_parent`) and
    crossovers (`num_crossovers`) of the current pareto frontier, and a new random subset
    of models (`num_random_mix`). The process is repeated until `num_iters` is reached.

    """

    def __init__(
        self,
        search_space: EvolutionarySearchSpace,
        search_objectives: SearchObjectives,
        output_dir: str,
        num_iters: Optional[int] = 10,
        init_num_models: Optional[int] = 10,
        initial_population_paths: Optional[List[str]] = None,
        num_random_mix: Optional[int] = 5,
        max_unseen_population: Optional[int] = 100,
        mutations_per_parent: Optional[int] = 1,
        num_crossovers: Optional[int] = 5,
        clear_evaluated_models: bool = True,
        save_pareto_model_weights: bool = True,
        seed: Optional[int] = 1,
    ):
        """Initialize the evolutionary search algorithm.

        Args:
            search_space: Discrete search space compatible with evolutionary algorithms.
            search_objectives: Search objectives.
            output_dir: Output directory.
            num_iters: Number of iterations.
            init_num_models: Number of initial models to evaluate.
            initial_population_paths: List of paths to the initial population of models.
                If `None`, `init_num_models` random models are used.
            num_random_mix: Number of random models to mix with the population in each iteration.
            max_unseen_population: Maximum number of unseen models to evaluate in each iteration.
            mutations_per_parent: Number of distinct mutations generated for each Pareto frontier member.
            num_crossovers: Total number of crossovers generated per iteration.
            clear_evaluated_models: Optimizes memory usage by clearing the architecture
                of `ArchaiModel` after each iteration. Defaults to True
            save_pareto_model_weights: If `True`, saves the weights of the pareto models. Defaults to True
            seed: Random seed.

        """

        assert isinstance(
            search_space, EvolutionarySearchSpace
        ), f"{str(search_space.__class__)} is not compatible with {str(self.__class__)}"

        self.iter_num = 0
        self.search_space = search_space
        self.so = search_objectives
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
        self.clear_evaluated_models = clear_evaluated_models
        self.save_pareto_model_weights = save_pareto_model_weights
        self.search_state = SearchResults(search_space, self.so)
        self.seed = seed
        self.rng = random.Random(seed)
        self.seen_archs = set()
        self.num_sampled_archs = 0

        assert self.init_num_models > 0
        assert self.num_iters > 0
        assert self.num_random_mix > 0
        assert self.max_unseen_population > 0

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
            valid_sample += [sample[i] for i in valid_indices]

        return valid_sample[:num_models]

    def mutate_parents(
        self, parents: List[ArchaiModel], mutations_per_parent: Optional[int] = 1, patience: Optional[int] = 20
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

        for p in tqdm(parents, desc="Mutating parents"):
            candidates = {}
            nb_tries = 0

            while len(candidates) < mutations_per_parent and nb_tries < patience:
                mutated_model = self.search_space.mutate(p)
                mutated_model.metadata["parent"] = p.archid

                if not self.so.is_model_valid(mutated_model):
                    continue

                if mutated_model.archid not in self.seen_archs:
                    mutated_model.metadata["generation"] = self.iter_num
                    candidates[mutated_model.archid] = mutated_model
                nb_tries += 1
            mutations.update(candidates)

        return list(mutations.values())

    def crossover_parents(
        self, parents: List[ArchaiModel], num_crossovers: Optional[int] = 1, patience: Optional[int] = 30
    ) -> List[ArchaiModel]:
        """Crossover parents to generate new models.

        Args:
            parents: List of parent models.
            num_crossovers: Number of crossovers to apply.
            patience: Number of tries to sample a valid model.

        Returns:
            List of crossovered models.

        """

        # Randomly samples k distinct pairs from `parents`
        children, children_ids = [], set()

        if len(parents) >= 2:
            pairs = [random.sample(parents, 2) for _ in range(num_crossovers)]
            for p1, p2 in pairs:
                child = self.search_space.crossover([p1, p2])
                nb_tries = 0

                while not self.so.is_model_valid(child) and nb_tries < patience:
                    child = self.search_space.crossover([p1, p2])
                    nb_tries += 1

                if child and self.so.is_model_valid(child):
                    if child.archid not in children_ids and child.archid not in self.seen_archs:
                        child.metadata["generation"] = self.iter_num
                        child.metadata["parents"] = f"{p1.archid},{p2.archid}"
                        children.append(child)
                        children_ids.add(child.archid)

        return children

    def on_calc_task_accuracy_end(self, current_pop: List[ArchaiModel]) -> None:
        """Callback function called right after calc_task_accuracy()."""

        pass

    def on_search_iteration_start(self, current_pop: List[ArchaiModel]) -> None:
        """Callback function called right before each search iteration."""

        pass

    def select_next_population(self, current_pop: List[ArchaiModel]) -> List[ArchaiModel]:
        """Select the next population from the current population

        Args:
            current_pop: Current population.

        Returns:
            Next population.

        """

        random.shuffle(current_pop)
        return current_pop[: self.max_unseen_population]

    @overrides
    def search(self) -> SearchResults:
        self.iter_num = 0

        if self.initial_population_paths:
            logger.info(f"Loading initial population from {len(self.initial_population_paths)} architectures ...")
            unseen_pop = [self.search_space.load_arch(path) for path in self.initial_population_paths]
        else:
            logger.info(f"Using {self.init_num_models} random architectures as the initial population ...")
            unseen_pop = self.sample_models(self.init_num_models)

        self.all_pop = unseen_pop

        for i in range(self.num_iters):
            self.iter_num = i + 1

            logger.info(f"Iteration {i+1}/{self.num_iters}")
            self.on_search_iteration_start(unseen_pop)

            # Calculates objectives
            logger.info(f"Calculating search objectives {list(self.so.objective_names)} for {len(unseen_pop)} models ...")

            results = self.so.eval_all_objs(unseen_pop)
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

            # Optimizes memory usage by clearing architectures from memory
            if self.clear_evaluated_models:
                logger.info("Optimzing memory usage ...")
                [model.clear() for model in unseen_pop]

            parents = pareto
            logger.info(f"Choosing {len(parents)} parents ...")

            # mutate random 'k' subsets of the parents
            # while ensuring the mutations fall within
            # desired constraint limits
            mutated = self.mutate_parents(parents, self.mutations_per_parent)
            logger.info(f"Mutation: {len(mutated)} new models.")

            # crossover random 'k' subsets of the parents
            # while ensuring the mutations fall within
            # desired constraint limits
            crossovered = self.crossover_parents(parents, self.num_crossovers)
            logger.info(f"Crossover: {len(crossovered)} new models.")

            # sample some random samples to add to the parent mix
            # to mitigage local minima
            rand_mix = self.sample_models(self.num_random_mix)
            unseen_pop = crossovered + mutated + rand_mix

            # shuffle before we pick a smaller population for the next stage
            logger.info(f"Total unseen population: {len(unseen_pop)}.")
            unseen_pop = self.select_next_population(unseen_pop)
            logger.info(f"Total unseen population after `max_unseen_population` restriction: {len(unseen_pop)}.")

            # update the set of architectures ever visited
            self.all_pop.extend(unseen_pop)

        return self.search_state
