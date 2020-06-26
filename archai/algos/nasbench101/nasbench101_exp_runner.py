# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Type

from overrides import overrides

from archai.nas.exp_runner import ExperimentRunner
from archai.nas.arch_trainer import ArchTrainer, TArchTrainer
from .petridish_cell_builder import PetridishCellBuilder

class PetridishExperimentRunner(ExperimentRunner):
    @overrides
    def cell_builder(self)->PetridishCellBuilder:
        return PetridishCellBuilder()

    @overrides
    def trainer_class(self)->TArchTrainer:
        return ArchTrainer

