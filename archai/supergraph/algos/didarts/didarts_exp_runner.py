# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Type

from overrides import overrides

from archai.supergraph.utils.nas.exp_runner import ExperimentRunner
from archai.supergraph.utils.nas.arch_trainer import TArchTrainer
from archai.supergraph.algos.darts.darts_model_desc_builder import DartsModelDescBuilder
from archai.supergraph.algos.didarts.didarts_arch_trainer import DidartsArchTrainer


class DiDartsExperimentRunner(ExperimentRunner):
    @overrides
    def model_desc_builder(self)->DartsModelDescBuilder:
        return DartsModelDescBuilder()

    @overrides
    def trainer_class(self)->TArchTrainer:
        return DidartsArchTrainer

