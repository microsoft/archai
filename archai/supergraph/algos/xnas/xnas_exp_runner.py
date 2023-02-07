# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from overrides import overrides

from archai.supergraph.algos.xnas.xnas_arch_trainer import XnasArchTrainer
from archai.supergraph.algos.xnas.xnas_model_desc_builder import XnasModelDescBuilder
from archai.supergraph.nas.arch_trainer import TArchTrainer
from archai.supergraph.nas.exp_runner import ExperimentRunner


class XnasExperimentRunner(ExperimentRunner):
    @overrides
    def model_desc_builder(self)->XnasModelDescBuilder:
        return XnasModelDescBuilder()

    @overrides
    def trainer_class(self)->TArchTrainer:
        return XnasArchTrainer

