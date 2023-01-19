# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from copy import deepcopy
from hashlib import sha1
from random import Random
from typing import Any, Dict, List, Optional

import torch
from overrides import overrides
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

from archai.api.archai_model import ArchaiModel
from archai.discrete_search.api.search_space import (
    BayesOptSearchSpace,
    EvolutionarySearchSpace,
)
from archai.discrete_search.search_spaces.nlp.transformer_flex.models.configuration_gpt2_flex import (
    GPT2FlexConfig,
)
from archai.discrete_search.search_spaces.nlp.transformer_flex.models.configuration_mem_transformer import (
    MemTransformerConfig,
)
from archai.discrete_search.search_spaces.nlp.transformer_flex.models.modeling_gpt2_flex import (
    GPT2FlexLMHeadModel,
)
from archai.discrete_search.search_spaces.nlp.transformer_flex.models.modeling_mem_transformer import (
    MemTransformerLMHeadModel,
)

# Register internal models to be compatible with auto classes
AutoConfig.register("gpt2-flex", GPT2FlexConfig)
AutoConfig.register("mem-transformer", MemTransformerConfig)

AutoModelForCausalLM.register(GPT2FlexConfig, GPT2FlexLMHeadModel)
AutoModelForCausalLM.register(MemTransformerConfig, MemTransformerLMHeadModel)


class TransformerFlexSearchSpace(EvolutionarySearchSpace, BayesOptSearchSpace):
    """Search space for Transformer models with flexible architecture.

    This class allows defining a search space for Transformer models with flexible architectures,
    using evolutionary or Bayesian optimization algorithms.

    The search space can be customized to include different values for hyperparameters,
    such as the number of layers, embedding dimensions, and number of attention heads.
    It also supports different Transformer variants, such as CodeGen, GPT-2, and Transformer-XL.

    """

    _DEFAULT_MODELS = {
        "codegen": {"d_model": "n_embd", "d_inner": "n_inner", "n_head": "n_head", "n_layer": "n_layer"},
        "gpt2": {
            "d_model": "n_embd",
            "d_inner": "n_inner",
            "n_head": "n_head",
            "n_layer": "n_layer",
            "vocab_size": "vocab_size",
            "max_sequence_length": "n_positions",
            "dropout": "resid_pdrop",
            "dropatt": "attn_pdrop",
        },
        "gpt2-flex": {
            "d_model": "n_embd",
            "d_inner": "n_inner",
            "n_head": "n_head",
            "n_layer": "n_layer",
            "vocab_size": "vocab_size",
            "max_sequence_length": "n_positions",
            "dropout": "resid_pdrop",
            "dropatt": "attn_pdrop",
        },
        "mem-transformer": {"d_model": "d_model", "d_inner": "d_inner", "n_head": "n_head", "n_layer": "n_layer"},
        "opt": {
            "d_model": "hidden_size",
            "d_inner": "ffn_dim",
            "n_head": "num_attention_heads",
            "n_layer": "num_hidden_layers",
        },
        "transfo-xl": {"d_model": "d_model", "d_inner": "d_inner", "n_head": "n_head", "n_layer": "n_layer"},
    }

    _DEFAULT_D_MODEL = list(range(128, 1024, 64))
    _DEFAULT_D_INNER = list(range(128, 1024, 64))
    _DEFAULT_N_HEAD = [2, 4, 8]

    def __init__(
        self,
        arch_type: str,
        min_layers: Optional[int] = 1,
        max_layers: Optional[int] = 10,
        d_inner_options: Optional[List[int]] = None,
        d_model_options: Optional[List[int]] = None,
        n_head_options: Optional[List[int]] = None,
        share_d_inner: Optional[bool] = True,
        mutation_prob: Optional[float] = 0.3,
        vocab_size: Optional[int] = 10_000,
        max_sequence_length: Optional[int] = 1024,
        att_dropout_rate: Optional[float] = 0.0,
        random_seed: Optional[int] = 1,
    ) -> None:
        """Initializes a `TransformerFlexSearchSpace` object.

        Args:
            arch_type: Type of Transformer architecture.
                Must be one of `gpt2`, `gpt2-flex`, `mem-transformer`, `opt`, `transfo-xl`.
            min_layers: Minimum number of layers in the model.
            max_layers: Maximum number of layers in the model.
            d_inner_options: List of options for the intermediate dimension (`d_inner`).
            d_model_options: List of options for the model dimension (`d_model`).
            n_head_options: List of options for the number of attention heads (`n_head`).
            share_d_inner: Whether to share the intermediate dimension (`d_inner`) across layers.
            mutation_prob: Probability of mutating a hyperparameter during evolution.
            vocab_size: Size of the vocabulary.
            max_sequence_length: Maximum sequence length.
            att_dropout_rate: Dropout rate for attention.
            random_seed: Random seed for reproducibility.

        """

        assert (
            arch_type in self._DEFAULT_MODELS
        ), f"The value of `arch_type` must be one of {list(self._DEFAULT_MODELS.keys())}"

        self.arch_type = arch_type

        self.min_layers = min_layers
        self.max_layers = max_layers

        self.options = {
            "d_inner": {"values": d_inner_options or self._DEFAULT_D_INNER, "share": share_d_inner},
            "d_model": {"values": d_model_options or self._DEFAULT_D_MODEL, "share": True},
            "n_head": {"values": n_head_options or self._DEFAULT_N_HEAD, "share": True},
        }

        self.mutation_prob = mutation_prob
        self.rng = Random(random_seed)

        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.att_dropout_rate = att_dropout_rate

    def _load_model_from_config(self, model_config: Dict[str, Any]) -> torch.nn.Module:
        """Loads a model from a configuration dictionary.

        Args:
            model_config: Configuration dictionary.

        Returns:
            A `torch.nn.Module` object.

        """

        param_map = self._DEFAULT_MODELS[self.arch_type]
        mapped_config = {param_map.get(p_name, p_name): p_value for p_name, p_value in model_config.items()}

        config = AutoConfig.for_model(self.arch_type, **mapped_config)
        return AutoModelForCausalLM.from_config(config)

    def get_archid(self, config: Dict[str, Any]) -> str:
        """Returns a unique identifier for a given configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            A unique identifier for the configuration.

        """

        pruned_config = deepcopy(config)
        n_layer = config["n_layer"]

        for c, opts in self.options.items():
            if not opts["share"]:
                pruned_config[c] = pruned_config[c][:n_layer]

        arch_str = json.dumps(pruned_config, sort_keys=True, ensure_ascii=True)
        return f'{self.arch_type}_{sha1(arch_str.encode("ascii")).hexdigest()}'

    @overrides
    def random_sample(self) -> ArchaiModel:
        model = None

        # Fixed params
        config = {
            "vocab_size": self.vocab_size,
            "dropatt": self.att_dropout_rate,
            "max_sequence_length": self.max_sequence_length,
        }

        while model is None:
            config["n_layer"] = self.rng.randint(self.min_layers, self.max_layers)

            for param, param_opts in self.options.items():
                if param_opts["share"]:
                    config[param] = self.rng.choice(param_opts["values"])
                else:
                    config[param] = [self.rng.choice(param_opts["values"]) for _ in range(self.max_layers)]

            if config["d_model"] % config["n_head"] == 0:
                model = self._load_model_from_config(config)

        return ArchaiModel(arch=model, archid=self.get_archid(config), metadata={"config": config})

    @overrides
    def save_arch(self, model: ArchaiModel, path: str) -> None:
        arch_config = model.metadata["config"]
        arch_config["arch_type"] = self.arch_type

        with open(path, "w", encoding="utf-8") as fp:
            json.dump(arch_config, fp, sort_keys=True, indent=2, ensure_ascii=True)

    @overrides
    def load_arch(self, path: str) -> ArchaiModel:
        with open(path, "r", encoding="utf-8") as fp:
            arch_config = json.load(fp)

        arch_type = arch_config.pop("arch_type")
        assert arch_type == self.arch_type, (
            f"Arch type value ({arch_type}) is different from the search space" f"arch type ({self.arch_type})."
        )

        return ArchaiModel(
            arch=self._load_model_from_config(arch_config),
            archid=self.get_archid(arch_config),
            metadata={"config": arch_config},
        )

    @overrides
    def save_model_weights(self, model: ArchaiModel, path: str) -> None:
        torch.save(model.arch.get_state_dict(), path)

    @overrides
    def load_model_weights(self, model: ArchaiModel, path: str) -> None:
        model.arch.load_state_dict(torch.load(path))

    @overrides
    def mutate(self, arch: ArchaiModel) -> ArchaiModel:
        config = deepcopy(arch.metadata["config"])

        if self.rng.random() < self.mutation_prob:
            config["n_layer"] = self.rng.randint(self.min_layers, self.max_layers)

        for param, opts in self.options.items():
            if opts["share"]:
                if self.rng.random() < self.mutation_prob:
                    config[param] = self.rng.choice(opts["values"])
            else:
                config[param] = [
                    self.rng.choice(opts["values"]) if self.rng.random() < self.mutation_prob else c
                    for c in config[param]
                ]

        return ArchaiModel(
            arch=self._load_model_from_config(config), archid=self.get_archid(config), metadata={"config": config}
        )

    @overrides
    def crossover(self, arch_list: List[ArchaiModel]) -> ArchaiModel:
        c0 = deepcopy(arch_list[0].metadata["config"])
        c1 = arch_list[1].metadata["config"]

        c0["n_layer"] = self.rng.choice([c0["n_layer"], c1["n_layer"]])

        for param, opts in self.options.items():
            if opts["share"]:
                c0[param] = self.rng.choice([c0[param], c1[param]])
            else:
                assert len(c0[param]) == len(c1[param]) == self.max_layers

                for layer in range(self.max_layers):
                    c0[param][layer] = self.rng.choice([c0[param][layer], c1[param][layer]])

        return ArchaiModel(arch=self._load_model_from_config(c0), archid=self.get_archid(c0), metadata={"config": c0})

    @overrides
    def encode(self, model: ArchaiModel) -> List[float]:
        config = model.metadata["config"]
        n_layer = config["n_layer"]
        gene = [n_layer]

        for param, opts in self.options.items():
            if opts["share"]:
                gene.append(config[param])
            else:
                gene += config[param][:n_layer]
                gene += [0] * (self.max_layers - n_layer)

        return gene
