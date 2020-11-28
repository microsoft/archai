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

from archai.common.common import get_expdir, logger

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

        # change relevant parts of conv_eval to ensure that freeze_evaler 
        # doesn't resume from checkpoints created by evaler and saves 
        # models to different names as well
        conf_eval['full_desc_filename'] = '$expdir/freeze_full_model_desc.yaml'
        conf_eval['metric_filename'] = '$expdir/freeze_eval_train_metrics.yaml'
        conf_eval['model_filename'] = '$expdir/freeze_model.pt'
        conf_eval['trainer']['epochs'] = conf_eval['trainer']['freeze_epochs']
        if conf_eval['checkpoint'] is not None:
            conf_eval['checkpoint']['filename'] = '$expdir/freeze_checkpoint.pth'

        logger.pushd('freeze_evaluate')
        freeze_evaler = FreezeEvaluator()
        freeze_eval_result = freeze_evaler.evaluate(conf_eval, model_desc_builder=self.model_desc_builder())
        logger.popd()

        # NOTE: Not returning freeze eval results to meet signature contract
        # but it seems like we don't need to anyways as everything we need is
        # logged to disk
        return reg_eval_result



