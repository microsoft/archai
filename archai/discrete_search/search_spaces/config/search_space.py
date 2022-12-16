from overrides import overrides
from random import Random
from typing import List, Type, Callable, Any

import numpy as np
import torch

from archai.discrete_search import (
    ArchaiModel, EvolutionarySearchSpace, BayesOptSearchSpace
)
from archai.discrete_search.search_spaces.config.arch_config import ArchConfig, build_arch_config
from archai.discrete_search.search_spaces.config.arch_param_tree import ArchParamTree
from archai.discrete_search.search_spaces.config import utils


def retry_on_exception(func: Callable[[], Any], num_retries: int) -> Any:
    exceptions = []

    for _ in range(num_retries):
        try:
            return func()
        except Exception as e:
            exceptions.append(e)

    raise exceptions[0]


class ConfigSearchSpace(EvolutionarySearchSpace, BayesOptSearchSpace):
    def __init__(self,
                 model_cls: Type[torch.nn.Module],
                 arch_param_tree: ArchParamTree,
                 seed: int = 1,
                 mutation_prob: float = 0.3,
                 track_unused_params: bool = True,
                 unused_param_value: int = 0, 
                 model_creation_attempts: int = 1,
                 **model_kwargs):
        self.model_cls = model_cls
        self.arch_param_tree = arch_param_tree
        self.mutation_prob = mutation_prob
        self.track_unused_params = track_unused_params
        self.unused_param_value = unused_param_value
        self.model_kwargs = model_kwargs
        self.model_creation_attempts = model_creation_attempts

        self.rng = Random(seed)

    def get_archid(self, arch_config: ArchConfig) -> str:
        e = self.arch_param_tree.encode_config(
            arch_config, track_unused_params=self.track_unused_params
        )
        return str(tuple(e))
    
    @overrides
    def save_arch(self, model: ArchaiModel, path: str) -> None:
        model.metadata['config'].to_json(path)
    
    @overrides
    def load_arch(self, path: str) -> ArchaiModel:
        config = ArchConfig.from_json(path)
        model = self.model_cls(config, **self.model_kwargs)
        
        return ArchaiModel(
            arch=model,
            archid=self.get_archid(config),
            metadata={'config': config}
        )
    
    @overrides
    def save_model_weights(self, model: ArchaiModel, path: str) -> None:
        torch.save(model.arch.get_state_dict(), path)
    
    @overrides
    def load_model_weights(self, model: ArchaiModel, path: str) -> None:
        model.arch.load_state_dict(torch.load(path))
        
    @overrides
    def random_sample(self) -> ArchaiModel:
        def _sample():
            config = self.arch_param_tree.sample_config(self.rng)
            model = self.model_cls(config, **self.model_kwargs)
            return model, config

        model, config = retry_on_exception(_sample, self.model_creation_attempts)

        return ArchaiModel(
            arch=model,
            archid=self.get_archid(config),
            metadata={'config': config}
        )

    @overrides
    def mutate(self, model: ArchaiModel) -> ArchaiModel:
        choices_dict = self.arch_param_tree.to_dict()

        def _mutate():
            # Mutates parameter with probability `self.mutation_prob`
            mutated_dict = utils.replace_ptree_pair_choices(
                choices_dict, 
                model.metadata['config'].to_dict(),
                lambda d_choice, current_choice: (
                    self.rng.choice(d_choice.choices)
                    if self.rng.random() < self.mutation_prob
                    else current_choice
                )
            )

            mutated_config = build_arch_config(mutated_dict)
            mutated_model = self.model_cls(mutated_config, **self.model_kwargs)

            return mutated_model, mutated_config
        
        mutated_model, mutated_config = retry_on_exception(_mutate, self.model_creation_attempts)
        
        return ArchaiModel(
            arch=mutated_model,
            archid=self.get_archid(mutated_config),
            metadata={'config': mutated_config}
        )

    @overrides
    def crossover(self, model_list: List[ArchaiModel]) -> ArchaiModel:
        def _crossover():
            # Selects two models from `model_list` to perform crossover
            model_1, model_2 = self.rng.choices(model_list, k=2)

            # Starting with arch param tree dict, randomly replaces DiscreteChoice objects
            # with params from model_1 with probability 0.5
            choices_dict = self.arch_param_tree.to_dict()
            cross_dict = utils.replace_ptree_pair_choices(
                choices_dict,
                model_1.metadata['config'].to_dict(),
                lambda d_choice, m1_value: (
                    m1_value
                    if self.rng.random() < 0.5
                    else d_choice
                )
            )

            # Replaces all remaining DiscreteChoice objects with params from model_2
            cross_dict = utils.replace_ptree_pair_choices(
                cross_dict,
                model_2.metadata['config'].to_dict(),
                lambda d_choice, m2_value: m2_value
            )

            cross_config = build_arch_config(cross_dict)
            cross_model = self.model_cls(cross_config, **self.model_kwargs)

            return cross_model, cross_config
        
        cross_model, cross_config = retry_on_exception(_crossover, self.model_creation_attempts)
        
        return ArchaiModel(
            arch=cross_model,
            archid=self.get_archid(cross_config),
            metadata={'config': cross_config}
        )

    @overrides
    def encode(self, model: ArchaiModel) -> np.ndarray:
        return np.array(self.arch_param_tree.encode_config(
            model.metadata['config'], 
            track_unused_params=self.track_unused_params
        ))
