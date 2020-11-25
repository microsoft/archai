# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from archai.nas.evaluater import EvalResult
from typing import Type

from overrides import overrides

from archai.common.config import Config
from archai.nas import nas_utils
from archai.nas.exp_runner import ExperimentRunner
from archai.nas.arch_trainer import ArchTrainer, TArchTrainer
from archai.nas.evaluater import Evaluater, EvalResult

from archai.algos.random.random_model_desc_builder import RandomModelDescBuilder
from .freeze_evaluator import FreezeEvaluator

class FreezeExperimentRunner(ExperimentRunner):
    @overrides
    def model_desc_builder(self)->RandomModelDescBuilder:
        return RandomModelDescBuilder()

    @overrides
    def trainer_class(self)->TArchTrainer:
        return None

    @overrides
    def run_eval(self, conf_eval:Config)->EvalResult:
        # regular evaluation of the architecture   
        evaler = self.evaluater()
        reg_eval_result = evaler.evaluate(conf_eval, model_desc_builder=self.model_desc_builder())

        freeze_evaler = FreezeEvaluator()
        freeze_eval_result = freeze_evaler.evaluate(conf_eval, model_desc_builder=self.model_desc_builder())

        # NOTE: Not returning freeze eval results
        # but it seems like we don't need to anyways as things get logged to disk
        return reg_eval_result



