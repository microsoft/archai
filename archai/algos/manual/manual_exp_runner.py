# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from typing import Optional, Type

from overrides import overrides

from archai.common.config import Config
from archai.nas import nas_utils
from archai.nas.exp_runner import ExperimentRunner
from archai.nas.arch_trainer import ArchTrainer, TArchTrainer
from archai.nas.model_desc_builder import ModelDescBuilder
from archai.algos.manual.manual_searcher import ManualSearcher
from archai.algos.manual.manual_evaluater import ManualEvaluater


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