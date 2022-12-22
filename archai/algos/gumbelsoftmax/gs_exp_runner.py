# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Type

from overrides import overrides

from archai.nas.exp_runner import ExperimentRunner
from archai.nas.arch_trainer import TArchTrainer
from archai.nas.finalizers import Finalizers
from archai.algos.gumbelsoftmax.gs_model_desc_builder import GsModelDescBuilder
from archai.algos.gumbelsoftmax.gs_arch_trainer import GsArchTrainer
from archai.algos.gumbelsoftmax.gs_finalizers import GsFinalizers

class GsExperimentRunner(ExperimentRunner):
    @overrides
    def model_desc_builder(self)->GsModelDescBuilder:
        return GsModelDescBuilder()

    @overrides
    def trainer_class(self)->TArchTrainer:
        return GsArchTrainer

    @overrides
    def finalizers(self)->Finalizers:
        return GsFinalizers()


