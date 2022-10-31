from overrides import overrides
from collections import OrderedDict
from random import Random
from copy import deepcopy
from typing import List, Type
import json

import numpy as np
import torch

from archai.discrete_search import (
    ArchaiModel, EvolutionarySearchSpace, BayesOptSearchSpace
)
from archai.discrete_search.search_spaces.builder import Cell, DiscreteChoice
from archai.discrete_search.search_spaces.builder.utils import (
    flatten_ordered_dict, 
    replace_param_tree_nodes,
    replace_param_tree_pair
)


class SearchSpaceBuilder(EvolutionarySearchSpace, BayesOptSearchSpace):
    def __init__(self, model_cls: Type[Cell], seed: int = 1,
                 mutation_prob: float = 0.3,
                 detect_unused_params: bool = True,
                 unused_param_value: int = 0, **model_kwargs):
        self.model_cls = model_cls
        self.mutation_prob = mutation_prob
        self.detect_unused_params = detect_unused_params
        self.unused_param_value = unused_param_value
        self.model_kwargs = model_kwargs

        self.search_param_tree = model_cls.get_search_params()
        self.rng = Random(seed)

    def get_archid(self, model: Cell):
        arch_params = flatten_ordered_dict(model._config)
        arch_tp = tuple(arch_params.values())

        if self.detect_unused_params:
            used_params = flatten_ordered_dict(model._used_params)
            assert used_params.keys() == arch_params.keys()

            arch_tp = tuple([
                param if used else self.unused_param_value
                for param, used in zip(arch_tp, used_params.values())
            ])
        
        return str(arch_tp)
    
    @overrides
    def save_arch(self, model: ArchaiModel, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as fp:
            json.dump(model.metadata['config'], fp)
    
    @overrides
    def load_arch(self, path: str) -> ArchaiModel:
        with open(path, encoding='utf-8') as fp:
            config = json.load(fp, object_pairs_hook_=OrderedDict)
        
        model = self.model_cls.from_config(config, **self.model_kwargs)
        
        return ArchaiModel(
            arch=model,
            archid=self.get_archid(model),
            metadata={'config': config}
        )
    
    @overrides
    def save_model_weights(self, model: ArchaiModel, path: str) -> None:
        torch.save(model.arch.get_state_dict(), path)
    
    @overrides
    def load_model_weights(self, model: ArchaiModel, path: str) -> None:
        model.arch.load_state_dict(torch.load(path))

    def _random_config(self, param_tree: OrderedDict) -> OrderedDict:
        return replace_param_tree_nodes(
            param_tree, lambda d: self.rng.choice(d.choices)
        )
        
    @overrides
    def random_sample(self) -> ArchaiModel:
        sampled_config = self._random_config(self.search_param_tree)
        model = self.model_cls.from_config(sampled_config, **self.model_kwargs)

        return ArchaiModel(
            arch=model,
            archid=self.get_archid(model),
            metadata={'config': sampled_config}
        )

    @overrides
    def mutate(self, model: ArchaiModel) -> ArchaiModel:
        mutated_config = replace_param_tree_pair(
            self.search_param_tree, 
            model.metadata['config'],
            lambda d_choice, param: (
                self.rng.choice(d_choice.choices)
                if self.rng.random() < self.mutation_prob
                else param
            )
        )

        model = self.model_cls.from_config(mutated_config, **self.model_kwargs)
        
        return ArchaiModel(
            arch=model,
            archid=self.get_archid(model),
            metadata={'config': mutated_config}
        )

    @overrides
    def crossover(self, model_list: List[ArchaiModel]) -> ArchaiModel:
        model_1, model_2 = self.rng.choices(model_list, k=2)

        cross_config = replace_param_tree_pair(
            model_1.metadata['config'],
            model_2.metadata['config'],
            lambda par1, par2: self.rng.choice([par1, par2])
        )

        model = self.model_cls.from_config(cross_config, **self.model_kwargs)
        
        return ArchaiModel(
            arch=model,
            archid=self.get_archid(model),
            metadata={'config': cross_config}
        )

    @overrides
    def encode(self, arch: ArchaiModel) -> np.ndarray:
        flt_conf = flatten_ordered_dict(arch.metadata['config'])
        return np.array([v for v in flt_conf.values()])
