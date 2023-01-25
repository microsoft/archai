# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import pytest
from transformers import GPT2LMHeadModel

from archai.discrete_search.api.search_space import (
    BayesOptSearchSpace,
    EvolutionarySearchSpace,
)
from archai.discrete_search.search_spaces.nlp.transformer_flex.search_space import (
    TransformerFlexSearchSpace,
)


@pytest.fixture
def config():
    return {
        "arch_type": "gpt2",
        "min_layers": 2,
        "max_layers": 3,
        "d_inner_options": [256, 512, 1024],
        "d_model_options": [256, 512, 1024],
        "n_head_options": [2, 4, 8],
        "share_d_inner": True,
        "mutation_prob": 1.0,
        "vocab_size": 50257,
        "max_sequence_length": 1024,
        "att_dropout_rate": 0.1,
        "random_seed": 42,
    }


def test_transformer_flex_search_space_init(config):
    search_space = TransformerFlexSearchSpace(**config)

    # Assert that the search space is set correctly
    assert search_space.arch_type == config["arch_type"]
    assert search_space.min_layers == config["min_layers"]
    assert search_space.max_layers == config["max_layers"]
    assert search_space.options["d_inner"]["values"] == config["d_inner_options"]
    assert search_space.options["d_model"]["values"] == config["d_model_options"]
    assert search_space.options["n_head"]["values"] == config["n_head_options"]
    assert search_space.options["d_inner"]["share"] == config["share_d_inner"]
    assert search_space.mutation_prob == config["mutation_prob"]
    assert search_space.vocab_size == config["vocab_size"]
    assert search_space.max_sequence_length == config["max_sequence_length"]
    assert search_space.att_dropout_rate == config["att_dropout_rate"]

    # Assert that the search space is a subclass of BayesOptSearchSpace
    # and EvolutionarySearchSpace
    assert isinstance(search_space, EvolutionarySearchSpace)
    assert isinstance(search_space, BayesOptSearchSpace)


def test_transformer_flex_search_space_load_model_from_config(config):
    # Assert that the model is loaded correctly
    search_space = TransformerFlexSearchSpace(**config)
    model_config = {"d_model": 256, "n_head": 2, "d_inner": 256, "n_layer": 2}
    model = search_space._load_model_from_config(model_config)
    assert isinstance(model, GPT2LMHeadModel)


def test_transformer_flex_search_space_get_archid(config):
    # Assert that the archid is generated correctly
    search_space = TransformerFlexSearchSpace(**config)
    model_config = {"d_model": 256, "n_head": 2, "d_inner": 256, "n_layer": 2}
    archid = search_space.get_archid(model_config)
    assert archid == "gpt2_9d72dac1ada7e094f5a7fd67dc688e33348d4907"


def test_transformer_flex_search_space_random_sample(config):
    # Assert that a model is sampled correctly
    search_space = TransformerFlexSearchSpace(**config)
    arch_model = search_space.random_sample()
    assert arch_model.archid == "gpt2_df9751a4db6ffaa963687eeae3f04d8c764f5f9c"
    assert isinstance(arch_model.arch, GPT2LMHeadModel)


def test_transformer_flex_search_space_save_arch(config):
    # Assert that a model is saved correctly
    search_space = TransformerFlexSearchSpace(**config)
    arch_model = search_space.random_sample()
    search_space.save_arch(arch_model, "test_arch.json")
    assert os.path.exists("test_arch.json")


def test_transformer_flex_search_space_load_arch(config):
    # Assert that a model is loaded correctly
    search_space = TransformerFlexSearchSpace(**config)
    arch_model = search_space.load_arch("test_arch.json")
    os.remove("test_arch.json")
    assert arch_model.archid == "gpt2_df9751a4db6ffaa963687eeae3f04d8c764f5f9c"


def test_transformer_flex_search_space_mutate(config):
    # Assert that a model is mutated correctly
    search_space = TransformerFlexSearchSpace(**config)
    arch_model = search_space.random_sample()
    mutated_arch_model = search_space.mutate(arch_model)
    assert mutated_arch_model.archid != arch_model.archid


def test_transformer_flex_search_space_crossover(config):
    search_space = TransformerFlexSearchSpace(**config)
    arch_model1 = search_space.random_sample()
    arch_model2 = search_space.random_sample()

    # Assert that a model is crossovered correctly
    crossovered_arch_model = search_space.crossover([arch_model1, arch_model2])
    assert crossovered_arch_model.archid != arch_model1.archid
    assert crossovered_arch_model.archid != arch_model2.archid


def test_transformer_flex_search_space_encode(config):
    # Assert that a model is encoded correctly
    search_space = TransformerFlexSearchSpace(**config)
    arch_model = search_space.random_sample()
    gene = search_space.encode(arch_model)
    assert gene == [2, 256, 1024, 4]
