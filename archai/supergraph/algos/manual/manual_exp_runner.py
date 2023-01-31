# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

from overrides import overrides

from archai.supergraph.algos.manual.manual_evaluater import ManualEvaluater
from archai.supergraph.algos.manual.manual_searcher import ManualSearcher
from archai.supergraph.nas.arch_trainer import TArchTrainer
from archai.supergraph.nas.exp_runner import ExperimentRunner
from archai.supergraph.nas.model_desc_builder import ModelDescBuilder


class ManualExperimentRunner(ExperimentRunner):
    """Runs manually designed models such as resnet"""

    @overrides
    def model_desc_builder(self)->Optional[ModelDescBuilder]:
        return None

    @overrides
    def trainer_class(self)->TArchTrainer:
        return None # no search trainer

    @overrides
    def searcher(self)->ManualSearcher:
        return ManualSearcher()

    @overrides
    def evaluater(self)->ManualEvaluater:
        return ManualEvaluater()

    @overrides
    def copy_search_to_eval(self)->None:
        pass