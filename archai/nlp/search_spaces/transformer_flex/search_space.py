from typing import List, Optional, Dict, Any
from random import Random
from copy import deepcopy
from hashlib import sha1
import json

import torch
from overrides import overrides

from archai.discrete_search import ArchaiModel
from archai.discrete_search import EvolutionarySearchSpace, BayesOptSearchSpace
from archai.nlp.search_spaces.transformer_flex.models.model_loader import (
    MODELS, load_model_from_config
)


class TransformerFlexSearchSpace(EvolutionarySearchSpace, BayesOptSearchSpace):
    _DEFAULT_D_MODEL = list(range(128, 1024, 64))
    _DEFAULT_D_INNER = list(range(128, 1024, 64))
    _DEFAULT_N_HEAD = [2, 4, 8]
    
    def __init__(self, arch_type: str, 
                 min_layers: int = 1,
                 max_layers: int = 10,
                 d_inner_options: Optional[List[int]] = None, 
                 d_model_options: Optional[List[int]] = None,
                 n_head_options: Optional[List[int]] = None,
                 share_d_inner: bool = True,
                 mutation_prob: float = 0.3,
                 random_seed: int = 1) -> None:

        assert arch_type in MODELS, \
            f'The value of `arch_type` must be one of {str(MODELS.keys())}'
        
        self.arch_type = arch_type
        
        self.min_layers = min_layers
        self.max_layers = max_layers
        
        self.options = {
            'd_inner': {
                'values': d_inner_options or self._DEFAULT_D_INNER,
                'share': share_d_inner
            },
            'd_model': {
                'values': d_model_options or self._DEFAULT_D_MODEL,
                'share': True
            },
            'n_head': {
                'values': n_head_options or self._DEFAULT_N_HEAD,
                'share': True
            }
        }

        self.mutation_prob = mutation_prob
        self.rng = Random(random_seed)

    def get_archid(self, config: Dict[str, Any]) -> str:
        pruned_config = deepcopy(config)
        n_layer = config['n_layer']

        for c, opts in self.options.items():
            if not opts['share']:
                pruned_config[c] = pruned_config[c][:n_layer]

        arch_str = json.dumps(pruned_config, sort_keys=True, ensure_ascii=True)
        return f'{self.arch_type}_{sha1(arch_str.encode("ascii")).hexdigest()}'

    @overrides
    def random_sample(self) -> ArchaiModel:
        config = {
            'n_layer': self.rng.randint(self.min_layers, self.max_layers)
        }

        for param, param_opts in self.options.items():
            if param_opts['share']:
                config[param] = self.rng.choice(param_opts['values'])
            else:
                config[param] = [
                    self.rng.choice(param_opts['values']) 
                    for _ in range(self.max_layers)
                ]

        return ArchaiModel(
            arch=load_model_from_config(self.arch_type, config),
            archid=self.get_archid(config),
            metadata={'config': config}
        )

    @overrides
    def save_arch(self, model: ArchaiModel, path: str) -> None:
        arch_config = model.metadata['config']
        arch_config['arch_type'] = self.arch_type
        
        with open(path, 'w', encoding='utf-8') as fp:
            json.dump(arch_config, fp, sort_keys=True, indent=2, ensure_ascii=True)
    
    @overrides
    def load_arch(self, path: str) -> ArchaiModel:
        with open(path, 'r', encoding='utf-8') as fp:
            arch_config = json.load(fp)
        
        arch_type = arch_config.pop('arch_type')
        
        return ArchaiModel(
            arch=load_model_from_config(arch_type, arch_config),
            archid=self.get_archid(arch_config),
            metadata={'config': arch_config}
        )

    @overrides
    def save_model_weights(self, model: ArchaiModel, path: str) -> None:
        torch.save(model.arch.get_state_dict(), path)

    @overrides
    def load_model_weights(self, model: ArchaiModel, path: str) -> None:
        model.arch.load_state_dict(torch.load(path))

    @overrides
    def mutate(self, arch: ArchaiModel) -> ArchaiModel:
        config = deepcopy(arch.metadata['config'])

        if self.rng.random() < self.mutation_prob:
            config['n_layer'] = self.rng.randint(self.min_layers, self.max_layers)
        
        for param, opts in self.options.items():
            if opts['share']:
                if self.rng.random() < self.mutation_prob:
                    config[param] = self.rng.choice(opts['values'])
            else:
                config[param] = [
                    self.rng.choice(opts['values']) if self.rng.random() < self.mutation_prob else c
                    for c in config[param]
                ]
            
        return ArchaiModel(
            arch=load_model_from_config(self.arch_type, config),
            archid=self.get_archid(config),
            metadata={'config': config}
        )

    @overrides
    def crossover(self, arch_list: List[ArchaiModel]) -> ArchaiModel:
        c0 = deepcopy(arch_list[0].metadata['config'])
        c1 = arch_list[1].metadata['config']

        c0['n_layer'] = self.rng.choice([c0['n_layer'], c1['n_layer']])
        
        for param, opts in self.options.items():
            if opts['share']:
                c0[param] = self.rng.choice([c0[param], c1[param]])
            else:
                assert len(c0[param]) == len(c1[param]) == self.max_layers

                for l in range(self.max_layers):
                    c0[param][l] = self.rng.choice([
                        c0[param][l], c1[param][l]
                    ])
        
        return ArchaiModel(
            arch=load_model_from_config(self.arch_type, c0),
            archid=self.get_archid(c0),
            metadata={'config': c0}
        )

    @overrides
    def encode(self, model: ArchaiModel) -> List[float]:
        config = model.metadata['config']
        n_layer = config['n_layer']
        gene = [n_layer]

        for param, opts in self.options.items():
            if opts['share']:
                gene.append(config[param])
            else:
                gene += config[param][:n_layer]
                gene += [0] * (self.max_layers - n_layer)

        return gene
