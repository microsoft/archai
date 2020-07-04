# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Type

from overrides import overrides

from archai.nas.exp_runner import ExperimentRunner
from archai.nas.arch_trainer import ArchTrainer, TArchTrainer
from .petridish_model_desc_builder import PetridishModelBuilder

class PetridishExperimentRunner(ExperimentRunner):
    @overrides
    def model_desc_builder(self)->PetridishModelBuilder:
        return PetridishModelBuilder()

    @overrides
    def trainer_class(self)->TArchTrainer:
        return ArchTrainer


    @overrides
    def _run_search(self, conf_search: Config) -> None:
        ''' Using special search class designed for petridish '''
        cell_builder = self.cell_builder()
        trainer_class = self.trainer_class()
        finalizers = self.finalizers()

        search = SearchDistributed(conf_search, cell_builder, trainer_class, finalizers)
        search.search_loop()
