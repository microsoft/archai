# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Type

from overrides import overrides

from archai.supergraph.nas.exp_runner import ExperimentRunner
from archai.supergraph.nas.arch_trainer import ArchTrainer, TArchTrainer
from archai.supergraph.algos.petridish.petridish_model_desc_builder import PetridishModelBuilder

class PetridishExperimentRunner(ExperimentRunner):
    @overrides
    def model_desc_builder(self)->PetridishModelBuilder:
        return PetridishModelBuilder()

    @overrides
    def trainer_class(self)->TArchTrainer:
        return ArchTrainer

