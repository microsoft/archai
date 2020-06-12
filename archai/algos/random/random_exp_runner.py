# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Type

from overrides import overrides

from archai.common.config import Config
from archai.nas import nas_utils
from archai.nas.exp_runner import ExperimentRunner
from archai.nas.arch_trainer import ArchTrainer, TArchTrainer
from .random_cell_builder import RandomCellBuilder

class RandomExperimentRunner(ExperimentRunner):
    @overrides
    def cell_builder(self)->RandomCellBuilder:
        return RandomCellBuilder()

    @overrides
    def trainer_class(self)->TArchTrainer:
        return None

