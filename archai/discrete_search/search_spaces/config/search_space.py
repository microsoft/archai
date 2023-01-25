# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from random import Random
from typing import List, Optional, Type

import numpy as np
import torch
from overrides import overrides

from archai.api.archai_model import ArchaiModel
from archai.discrete_search.api.search_space import (
    BayesOptSearchSpace,
    EvolutionarySearchSpace,
)
from archai.discrete_search.search_spaces.config import utils
from archai.discrete_search.search_spaces.config.arch_config import (
    ArchConfig,
    build_arch_config,
)
from archai.discrete_search.search_spaces.config.arch_param_tree import ArchParamTree


class ConfigSearchSpace(EvolutionarySearchSpace, BayesOptSearchSpace):
    """Search space for discrete search space with a configuration file."""

    def __init__(
        self,
        model_cls: Type[torch.nn.Module],
        arch_param_tree: ArchParamTree,
        seed: Optional[int] = 1,
        mutation_prob: Optional[float] = 0.3,
        track_unused_params: Optional[bool] = True,
        unused_param_value: Optional[int] = 0,
        **model_kwargs
    ) -> None:
        """Initialize the search space.

        Args:
            model_cls: Model class.
            arch_param_tree: Architecture parameter tree.
            seed: Random seed.
            mutation_prob: Mutation probability.
            track_unused_params: Whether to track unused parameters.
            unused_param_value: Value to use for unused parameters.

        """

        self.model_cls = model_cls
        self.arch_param_tree = arch_param_tree
        self.mutation_prob = mutation_prob
        self.track_unused_params = track_unused_params
        self.unused_param_value = unused_param_value
        self.model_kwargs = model_kwargs

        self.rng = Random(seed)

    def get_archid(self, arch_config: ArchConfig) -> str:
        """Return the architecture identifier for the given architecture configuration.

        Args:
            arch_config: Architecture configuration.

        Returns:
            Architecture identifier.

        """

        e = self.arch_param_tree.encode_config(arch_config, track_unused_params=self.track_unused_params)
        return str(tuple(e))

    @overrides
    def save_arch(self, model: ArchaiModel, path: str) -> None:
        model.metadata["config"].to_file(path)

    @overrides
    def load_arch(self, path: str) -> ArchaiModel:
        config = ArchConfig.from_file(path)
        model = self.model_cls(config, **self.model_kwargs)

        return ArchaiModel(arch=model, archid=self.get_archid(config), metadata={"config": config})

    @overrides
    def save_model_weights(self, model: ArchaiModel, path: str) -> None:
        torch.save(model.arch.get_state_dict(), path)

    @overrides
    def load_model_weights(self, model: ArchaiModel, path: str) -> None:
        model.arch.load_state_dict(torch.load(path))

    @overrides
    def random_sample(self) -> ArchaiModel:
        config = self.arch_param_tree.sample_config(self.rng)
        model = self.model_cls(config, **self.model_kwargs)

        return ArchaiModel(arch=model, archid=self.get_archid(config), metadata={"config": config})

    @overrides
    def mutate(self, model: ArchaiModel) -> ArchaiModel:
        choices_dict = self.arch_param_tree.to_dict()

        # Mutates parameter with probability `self.mutation_prob`
        mutated_dict = utils.replace_ptree_pair_choices(
            choices_dict,
            model.metadata["config"].to_dict(),
            lambda d_choice, current_choice: (
                self.rng.choice(d_choice.choices) if self.rng.random() < self.mutation_prob else current_choice
            ),
        )

        mutated_config = build_arch_config(mutated_dict)
        mutated_model = self.model_cls(mutated_config, **self.model_kwargs)

        return ArchaiModel(
            arch=mutated_model, archid=self.get_archid(mutated_config), metadata={"config": mutated_config}
        )

    @overrides
    def crossover(self, model_list: List[ArchaiModel]) -> ArchaiModel:
        # Selects two models from `model_list` to perform crossover
        model_1, model_2 = self.rng.choices(model_list, k=2)

        # Starting with arch param tree dict, randomly replaces DiscreteChoice objects
        # with params from model_1 with probability 0.5
        choices_dict = self.arch_param_tree.to_dict()
        cross_dict = utils.replace_ptree_pair_choices(
            choices_dict,
            model_1.metadata["config"].to_dict(),
            lambda d_choice, m1_value: (m1_value if self.rng.random() < 0.5 else d_choice),
        )

        # Replaces all remaining DiscreteChoice objects with params from model_2
        cross_dict = utils.replace_ptree_pair_choices(
            cross_dict, model_2.metadata["config"].to_dict(), lambda d_choice, m2_value: m2_value
        )

        cross_config = build_arch_config(cross_dict)
        cross_model = self.model_cls(cross_config, **self.model_kwargs)

        return ArchaiModel(arch=cross_model, archid=self.get_archid(cross_config), metadata={"config": cross_config})

    @overrides
    def encode(self, model: ArchaiModel) -> np.ndarray:
        return np.array(
            self.arch_param_tree.encode_config(model.metadata["config"], track_unused_params=self.track_unused_params)
        )
