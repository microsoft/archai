# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from archai.nas.evaluater import EvalResult
from typing import Type
from copy import deepcopy

from overrides import overrides

from archai.common.config import Config
from archai.nas import nas_utils
from archai.nas.exp_runner import ExperimentRunner
from archai.nas.arch_trainer import ArchTrainer, TArchTrainer
from archai.nas.evaluater import Evaluater, EvalResult

from archai.common.common import get_expdir, logger

from archai.algos.random_sample_darts_space.random_model_desc_builder import RandomModelDescBuilder
from archai.algos.proxynas.freeze_manual_searcher import ManualFreezeSearcher
from ..random_sample_darts_space.darts_space_evaluater import DartsSpaceEvaluater
from .freeze_darts_space_evaluater import FreezeDartsSpaceEvaluater

class FreezeDartsSpaceExperimentRunner(ExperimentRunner):
    ''' Samples a reproducible random architecture from DARTS search space and freeze trains it '''
    
    @overrides
    def model_desc_builder(self)->RandomModelDescBuilder:
        return RandomModelDescBuilder()

    @overrides
    def trainer_class(self)->TArchTrainer:
        return None

    @overrides
    def searcher(self)->ManualFreezeSearcher:
        return ManualFreezeSearcher() # no searcher basically

    @overrides
    def copy_search_to_eval(self)->None:
        pass

    @overrides
    def run_eval(self, conf_eval:Config)->EvalResult:
        
        # regular evaluation of the architecture
        # this is expensive 
        # --------------------------------------
        logger.pushd('regular_evaluate')
        reg_eval_result = None
        if conf_eval['trainer']['train_regular']:
            evaler = DartsSpaceEvaluater()
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
        if conf_eval['trainer']['train_fear']:
            freeze_evaler = FreezeDartsSpaceEvaluater()
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