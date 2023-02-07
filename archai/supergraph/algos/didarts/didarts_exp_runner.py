# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from overrides import overrides

from archai.supergraph.algos.darts.darts_model_desc_builder import DartsModelDescBuilder
from archai.supergraph.algos.didarts.didarts_arch_trainer import DidartsArchTrainer
from archai.supergraph.nas.arch_trainer import TArchTrainer
from archai.supergraph.nas.exp_runner import ExperimentRunner


class DiDartsExperimentRunner(ExperimentRunner):
    @overrides
    def model_desc_builder(self)->DartsModelDescBuilder:
        return DartsModelDescBuilder()

    @overrides
    def trainer_class(self)->TArchTrainer:
        return DidartsArchTrainer

