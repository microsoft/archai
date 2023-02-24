
from overrides import overrides
import json
import numpy as np
from random import Random
import torch
from typing import List

from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.search_space import EvolutionarySearchSpace

from model import MyModel

class CNNSearchSpace(EvolutionarySearchSpace):
    def __init__(self, min_layers: int = 1, max_layers: int = 12,
                 kernel_list=(1, 3, 5, 7), hidden_list=(16, 32, 64, 128),
                 seed: int = 1):

        self.min_layers = min_layers
        self.max_layers = max_layers
        self.kernel_list = kernel_list
        self.hidden_list = hidden_list

        self.rng = Random(seed)

    @overrides
    def random_sample(self) -> ArchaiModel:
        # Randomly chooses architecture parameters
        nb_layers = self.rng.randint(self.min_layers, self.max_layers)
        kernel_size = self.rng.choice(self.kernel_list)
        hidden_dim = self.rng.choice(self.hidden_list)

        model = MyModel(nb_layers, kernel_size, hidden_dim)

        # Wraps model into ArchaiModel
        return ArchaiModel(arch=model, archid=model.get_archid())

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

        return ArchaiModel(arch=model, archid=model.get_archid())

    @overrides
    def save_model_weights(self, model: ArchaiModel, file: str):
        state_dict = model.arch.get_state_dict()
        torch.save(state_dict, file)

    @overrides
    def load_model_weights(self, model: ArchaiModel, file: str):
        model.arch.load_state_dict(torch.load(file))

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
            arch=mutated_model, archid=mutated_model.get_archid()
        )

    @overrides
    def crossover(self, model_list: List[ArchaiModel]) -> ArchaiModel:
        model_1, model_2 = model_list[:2]

        new_config = {
            'nb_layers': self.rng.choice([model_1.arch.nb_layers, model_2.arch.nb_layers]),
            'kernel_size': self.rng.choice([model_1.arch.kernel_size, model_2.arch.kernel_size]),
            'hidden_dim': self.rng.choice([model_1.arch.hidden_dim, model_2.arch.hidden_dim]),
        }

        crossover_model = MyModel(**new_config)

        return ArchaiModel(
            arch=crossover_model, archid=crossover_model.get_archid()
        )

    @overrides
    def encode(self, model: ArchaiModel) -> np.ndarray:
        return np.array([model.arch.nb_layers, model.arch.kernel_size, model.arch.hidden_dim])