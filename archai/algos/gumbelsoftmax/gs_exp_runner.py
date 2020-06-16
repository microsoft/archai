# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Type

from overrides import overrides

from archai.nas.exp_runner import ExperimentRunner
from archai.nas.arch_trainer import TArchTrainer
from archai.nas.finalizers import Finalizers
from .gs_cell_builder import GsCellBuilder
from .gs_arch_trainer import GsArchTrainer
from .gs_finalizers import GsFinalizers

class GsExperimentRunner(ExperimentRunner):
    @overrides
    def cell_builder(self)->GsCellBuilder:
        return GsCellBuilder()

    @overrides
    def trainer_class(self)->TArchTrainer:
        return GsArchTrainer

    @overrides
    def finalizers(self)->Finalizers:
        return GsFinalizers()
        
        
