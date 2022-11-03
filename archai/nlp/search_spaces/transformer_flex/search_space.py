# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Transformer-Flex Search Space.
"""

from typing import List, Optional, Dict, Any
from random import Random
from copy import deepcopy
from hashlib import sha1
import json

import torch
from overrides import overrides

from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.configuration_auto import AutoConfig

from archai.discrete_search import ArchaiModel
from archai.discrete_search import EvolutionarySearchSpace, BayesOptSearchSpace


class TransformerFlexSearchSpace(EvolutionarySearchSpace, BayesOptSearchSpace):
    _DEFAULT_MODELS = {
        "codegen": {
            "d_model": "n_ctx",
            "d_inner": "n_inner",
            "n_head": "n_head",
            "n_layer": "n_layer"
        },
        "gpt2": {
            "d_model": "n_embd",
            "d_inner": "n_inner",
            "n_head": "n_head",
            "n_layer": "n_layer"
        },
        "opt": {
            "d_model": "hidden_size",
            "d_inner": "ffn_dim",
            "n_head": "num_attention_heads",
            "n_layer": "num_hidden_layers"
        },
        "transfo_xl": {
            "d_model": "d_model",
            "d_inner": "d_inner",
            "n_head": "n_head",
            "n_layer": "n_layer"
        }
    }

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

        assert arch_type in self._DEFAULT_MODELS, \
            f'The value of `arch_type` must be one of {list(self._DEFAULT_MODELS.keys())}'
        
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

    def _load_model_from_config(self, model_config: Dict[str, Any]) -> torch.nn.Module:
        model_config = {
            new_k: model_config[k] 
            for k, new_k in self._DEFAULT_MODELS[self.arch_type].items()
        }

        config = AutoConfig.for_model(self.arch_type, **model_config)
        return AutoModelForCausalLM.from_config(config)

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
        model = None

        while model is None:
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

            if config['d_model'] % config['n_head'] == 0:
                model = self._load_model_from_config(config)
        
        return ArchaiModel(
            arch=model,
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
            arch=self._load_model_from_config(arch_type, arch_config),
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
            arch=self._load_model_from_config(config),
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
            arch=self._load_model_from_config(self.arch_type, c0),
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
