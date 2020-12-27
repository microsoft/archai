# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from typing import Optional, Type

from overrides import overrides
from copy import deepcopy

from archai.common.config import Config
from archai.nas import nas_utils
from archai.nas.exp_runner import ExperimentRunner
from archai.nas.arch_trainer import ArchTrainer, TArchTrainer
from archai.nas.model_desc_builder import ModelDescBuilder
from archai.nas.evaluater import EvalResult
from .freeze_manual_evaluater import ManualFreezeEvaluater
from .freeze_manual_searcher import ManualFreezeSearcher

from archai.algos.naswotrain.naswotrain_manual_evaluator import NaswotrainManualEvaluater

from archai.common.common import get_expdir, logger

class ManualFreezeExperimentRunner(ExperimentRunner):
    """Runs manually designed models such as resnet"""

    @overrides
    def model_desc_builder(self)->Optional[ModelDescBuilder]:
        return None

    @overrides
    def trainer_class(self)->TArchTrainer:
        return None # no search trainer

    @overrides
    def searcher(self)->ManualFreezeSearcher:
        return ManualFreezeSearcher()

    @overrides
    def evaluater(self)->ManualFreezeEvaluater:
        return ManualFreezeEvaluater()

    @overrides
    def copy_search_to_eval(self)->None:
        pass

    @overrides
    def run_eval(self, conf_eval:Config)->EvalResult:
        # without training architecture evaluation score
        # ---------------------------------------
        logger.pushd('naswotrain_evaluate')
        naswotrain_evaler = NaswotrainManualEvaluater()
        conf_eval_naswotrain = deepcopy(conf_eval)

        if conf_eval_naswotrain['checkpoint'] is not None:
            conf_eval_naswotrain['checkpoint']['filename'] = '$expdir/naswotrain_checkpoint.pth'

        naswotrain_eval_result = naswotrain_evaler.evaluate(conf_eval_naswotrain, model_desc_builder=self.model_desc_builder())
        logger.popd()

        # regular evaluation of the architecture
        # this is expensive 
        # --------------------------------------
        logger.pushd('regular_evaluate')
        reg_eval_result = None
        if conf_eval['trainer']['train_regular']:
            evaler = self.evaluater()
            conf_eval_reg = deepcopy(conf_eval)
            reg_eval_result = evaler.evaluate(conf_eval_reg, model_desc_builder=self.model_desc_builder())
        logger.popd()

        # freeze train evaluation of the architecture
        # -------------------------------------------
        # change relevant parts of conv_eval to ensure that freeze_evaler 
        # doesn't resume from checkpoints created by evaler and saves 
        # models to different names as well
        conf_eval['full_desc_filename'] = '$expdir/freeze_full_model_desc.yaml'
        conf_eval['metric_filename'] = '$expdir/freeze_eval_train_metrics.yaml'
        conf_eval['model_filename'] = '$expdir/freeze_model.pt'
        
        if conf_eval['checkpoint'] is not None:
            conf_eval['checkpoint']['filename'] = '$expdir/freeze_checkpoint.pth'

        logger.pushd('freeze_evaluate')
        freeze_evaler = ManualFreezeEvaluater()
        conf_eval_freeze = deepcopy(conf_eval)
        freeze_eval_result = freeze_evaler.evaluate(conf_eval_freeze, model_desc_builder=self.model_desc_builder())
        logger.popd()

        # NOTE: Not returning freeze eval results to meet signature contract
        # but it seems like we don't need to anyways as everything we need is
        # logged to disk
        if reg_eval_result is not None:
            return reg_eval_result
        else:
            return freeze_eval_result    
