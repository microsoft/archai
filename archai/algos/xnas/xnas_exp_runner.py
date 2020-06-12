# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Type

from overrides import overrides

from archai.nas.exp_runner import ExperimentRunner
from archai.nas.arch_trainer import TArchTrainer
from .xnas_cell_builder import XnasCellBuilder
from .xnas_arch_trainer import XnasArchTrainer

class XnasExperimentRunner(ExperimentRunner):
    @overrides
    def cell_builder(self)->XnasCellBuilder:
        return XnasCellBuilder()

    @overrides
    def trainer_class(self)->TArchTrainer:
        return XnasArchTrainer

