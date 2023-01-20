# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Type

from overrides import overrides

from archai.common.config import Config
from archai.supergraph.utils.nas import nas_utils
from archai.supergraph.utils.nas.exp_runner import ExperimentRunner
from archai.supergraph.utils.nas.arch_trainer import ArchTrainer, TArchTrainer
from archai.supergraph.algos.random.random_model_desc_builder import RandomModelDescBuilder

class RandomExperimentRunner(ExperimentRunner):
    @overrides
    def model_desc_builder(self)->RandomModelDescBuilder:
        return RandomModelDescBuilder()

    @overrides
    def trainer_class(self)->TArchTrainer:
        return None

