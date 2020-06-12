# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Type

from overrides import overrides

from archai.nas.exp_runner import ExperimentRunner
from archai.nas.arch_trainer import TArchTrainer
from archai.algos.darts.darts_cell_builder import DartsCellBuilder
from archai.algos.didarts.didarts_arch_trainer import DidartsArchTrainer


class DiDartsExperimentRunner(ExperimentRunner):
    @overrides
    def cell_builder(self)->DartsCellBuilder:
        return DartsCellBuilder()

    @overrides
    def trainer_class(self)->TArchTrainer:
        return DidartsArchTrainer

