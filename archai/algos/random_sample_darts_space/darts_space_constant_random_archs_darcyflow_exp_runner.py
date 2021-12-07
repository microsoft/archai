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
from archai.algos.random_sample_darts_space.darts_space_darcyflow_evaluater import DartsSpaceDarcyflowEvaluater

class DartsSpaceConstantRandomArchsDarcyflowExpRunner(ExperimentRunner):
    ''' Samples a reproducible random architecture from
    DARTS search space and trains it but is specific to DarcyFlow dataset 
    since it is a 2D input, dense 2D output task '''
    
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
        evaler = DartsSpaceDarcyflowEvaluater()
        conf_eval_reg = deepcopy(conf_eval)
        reg_eval_result = evaler.evaluate(conf_eval_reg, model_desc_builder=self.model_desc_builder())
        logger.popd()

        return reg_eval_result
