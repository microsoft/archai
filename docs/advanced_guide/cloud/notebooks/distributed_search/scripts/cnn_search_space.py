
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

    def get_model_json(self, archid: str) -> ArchaiModel:
        model = MyModel.from_archid(archid)
        return {
                'nb_layers': model.nb_layers,
                'kernel_size': model.kernel_size,
                'hidden_dim': model.hidden_dim
            }

    @overrides
    def save_arch(self, model: ArchaiModel, file: str):
        config = self.get_model_json(model.archid)
        with open(file, 'w') as fp:
            json.dump(config, fp)

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
        config = self.get_model_json(model_1.archid)
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
        arch_1 = MyModel.from_archid(model_1.archid)
        arch_2 = MyModel.from_archid(model_2.archid)
        new_config = {
            'nb_layers': self.rng.choice([arch_1.nb_layers, arch_2.nb_layers]),
            'kernel_size': self.rng.choice([arch_1.kernel_size, arch_2.kernel_size]),
            'hidden_dim': self.rng.choice([arch_1.hidden_dim, arch_2.hidden_dim]),
        }

        crossover_model = MyModel(**new_config)

        return ArchaiModel(
            arch=crossover_model, archid=crossover_model.get_archid()
        )


if __name__ == "__main__":
    space = CNNSearchSpace()
    m = space.random_sample()
    print(m.archid)
    space.save_arch(m, 'test.json')
    m2 = space.load_arch('test.json')
    assert(m.archid == m2.archid)
    m3 = space.random_sample()
    print(m3.archid)
    m4 = space.mutate(m2)
    print(m4.archid)
    m5 = space.crossover([m2, m3])
    print(m5.archid)
