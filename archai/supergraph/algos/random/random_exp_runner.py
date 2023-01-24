# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from overrides import overrides

from archai.supergraph.nas.exp_runner import ExperimentRunner
from archai.supergraph.nas.arch_trainer import TArchTrainer
from archai.supergraph.algos.random.random_model_desc_builder import RandomModelDescBuilder

class RandomExperimentRunner(ExperimentRunner):
    @overrides
    def model_desc_builder(self)->RandomModelDescBuilder:
        return RandomModelDescBuilder()

    @overrides
    def trainer_class(self)->TArchTrainer:
        return None

