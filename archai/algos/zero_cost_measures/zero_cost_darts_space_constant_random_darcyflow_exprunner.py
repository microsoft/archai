# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from typing import Optional, Type
from copy import deepcopy
import os

from overrides import overrides

from archai.common.config import Config
from archai.nas import nas_utils
from archai.common import utils
from archai.nas.exp_runner import ExperimentRunner
from archai.nas.arch_trainer import ArchTrainer, TArchTrainer
from archai.nas.model_desc_builder import ModelDescBuilder
from archai.nas.evaluater import EvalResult
from archai.common.common import get_expdir, logger
from archai.algos.proxynas.freeze_manual_searcher import ManualFreezeSearcher
from archai.algos.zero_cost_measures.zero_cost_darts_space_constant_random_evaluator import ZeroCostDartsSpaceConstantRandomEvaluator
from archai.algos.random_sample_darts_space.random_model_desc_builder import RandomModelDescBuilder

from nats_bench import create

class ZeroCostDartsSpaceConstantRandomDarcyFlowExpRunner(ExperimentRunner):
    """Runs zero cost on architectures from DARTS search space 
    which are randomly sampled in a reproducible way"""

    @overrides
    def model_desc_builder(self)->RandomModelDescBuilder:
        return RandomModelDescBuilder()

    @overrides
    def trainer_class(self)->TArchTrainer:
        return None # no search trainer

    @overrides
    def searcher(self)->ManualFreezeSearcher:
        return ManualFreezeSearcher() # no searcher basically

    @overrides
    def copy_search_to_eval(self)->None:
        pass

    @overrides
    def run_eval(self, conf_eval:Config)->EvalResult:
        # without training architecture evaluation score
        # ---------------------------------------
        logger.pushd('zerocost_evaluate')
        zerocost_evaler = ZeroCostDartsSpaceConstantRandomEvaluator()
        conf_eval_zerocost = deepcopy(conf_eval)

        if conf_eval_zerocost['checkpoint'] is not None:
            conf_eval_zerocost['checkpoint']['filename'] = '$expdir/zerocost_checkpoint.pth'

        zerocost_eval_result = zerocost_evaler.evaluate(conf_eval_zerocost, model_desc_builder=self.model_desc_builder())
        logger.popd()

        return zerocost_eval_result