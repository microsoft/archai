from typing import List, Optional

from overrides import overrides
import numpy as np
import torch
from torch import nn
from archai.discrete_search.api import ArchaiModel

import json
from random import Random
from archai.discrete_search.api import DiscreteSearchSpace
from model import MyModel


class CNNSearchSpace(DiscreteSearchSpace):
    def __init__(self, min_layers: int = 1, max_layers: int = 12,
                 kernel_list=(1, 3, 5, 7), hidden_list=(16, 32, 64, 128),
                 seed: int = 1):

        self.min_layers = min_layers
        self.max_layers = max_layers
        self.kernel_list = kernel_list
        self.hidden_list = hidden_list

        self.rng = Random(seed)

    def get_archid(self, model: MyModel) -> str:
        return f'L={model.nb_layers}, K={model.kernel_size}, H={model.hidden_dim}'

    @overrides
    def random_sample(self) -> ArchaiModel:
        # Randomly chooses architecture parameters
        nb_layers = self.rng.randint(self.min_layers, self.max_layers)
        kernel_size = self.rng.choice(self.kernel_list)
        hidden_dim = self.rng.choice(self.hidden_list)

        model = MyModel(nb_layers, kernel_size, hidden_dim)

        # Wraps model into ArchaiModel
        return ArchaiModel(arch=model, archid=self.get_archid(model))

    @overrides
    def save_arch(self, model: ArchaiModel, file: str):
        with open(file, 'w') as fp:
            json.dump({
                'nb_layers': model.arch.nb_layers,
                'kernel_size': model.arch.kernel_size,
                'hidden_dim': model.arch.hidden_dim
            }, fp)

    @overrides
    def load_arch(self, file: str):
        config = json.load(open(file))
        model = MyModel(**config)

        return ArchaiModel(arch=model, archid=self.get_archid(model))

    @overrides
    def save_model_weights(self, model: ArchaiModel, file: str):
        state_dict = model.arch.get_state_dict()
        torch.save(state_dict, file)

    @overrides
    def load_model_weights(self, model: ArchaiModel, file: str):
        model.arch.load_state_dict(torch.load(file))


from archai.discrete_search.api.search_space import EvolutionarySearchSpace, BayesOptSearchSpace

class CNNSearchSpaceExt(CNNSearchSpace, EvolutionarySearchSpace, BayesOptSearchSpace):
    ''' We are subclassing CNNSearchSpace just to save up space'''

    @overrides
    def mutate(self, model_1: ArchaiModel) -> ArchaiModel:
        config = {
            'nb_layers': model_1.arch.nb_layers,
            'kernel_size': model_1.arch.kernel_size,
            'hidden_dim': model_1.arch.hidden_dim
        }

        if self.rng.random() < 0.2:
            config['nb_layers'] = self.rng.randint(self.min_layers, self.max_layers)

        if self.rng.random() < 0.2:
            config['kernel_size'] = self.rng.choice(self.kernel_list)

        if self.rng.random() < 0.2:
            config['hidden_dim'] = self.rng.choice(self.hidden_list)

        mutated_model = MyModel(**config)

        return ArchaiModel(
            arch=mutated_model, archid=self.get_archid(mutated_model)
        )

    @overrides
    def crossover(self, model_list: List[ArchaiModel]) -> ArchaiModel:
        new_config = {
            'nb_layers': self.rng.choice([m.arch.nb_layers for m in model_list]),
            'kernel_size': self.rng.choice([m.arch.kernel_size for m in model_list]),
            'hidden_dim': self.rng.choice([m.arch.hidden_dim for m in model_list]),
        }

        crossover_model = MyModel(**new_config)

        return ArchaiModel(
            arch=crossover_model, archid=self.get_archid(crossover_model)
        )

    @overrides
    def encode(self, model: ArchaiModel) -> np.ndarray:
        return np.array([model.arch.nb_layers, model.arch.kernel_size, model.arch.hidden_dim])