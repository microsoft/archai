# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from overrides import overrides

from archai.supergraph.nas.exp_runner import ExperimentRunner
from archai.supergraph.nas.arch_trainer import TArchTrainer
from archai.supergraph.algos.darts.darts_model_desc_builder import DartsModelDescBuilder
from archai.supergraph.algos.darts.bilevel_arch_trainer import BilevelArchTrainer


class DartsExperimentRunner(ExperimentRunner):
    @overrides
    def model_desc_builder(self)->DartsModelDescBuilder:
        return DartsModelDescBuilder()

    @overrides
    def trainer_class(self)->TArchTrainer:
        return BilevelArchTrainer

