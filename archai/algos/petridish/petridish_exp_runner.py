# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Type

from overrides import overrides

from archai.nas.exp_runner import ExperimentRunner
from archai.nas.arch_trainer import ArchTrainer, TArchTrainer
from archai.nas.search_distributed import SearchDistributed
from archai.common.config import Config
from .petridish_cell_builder import PetridishCellBuilder

class PetridishExperimentRunner(ExperimentRunner):
    @overrides
    def cell_builder(self)->PetridishCellBuilder:
        return PetridishCellBuilder()

    @overrides
    def trainer_class(self)->TArchTrainer:
        return ArchTrainer


    # @overrides
    # def _run_search(self, conf_search: Config) -> None:
    #     ''' Using special search class designed for petridish '''
    #     cell_builder = self.cell_builder()
    #     trainer_class = self.trainer_class()
    #     finalizers = self.finalizers()

    #     search = SearchDistributed(conf_search, cell_builder, trainer_class, finalizers)
    #     search.search_loop()

