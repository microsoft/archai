# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from archai.algos.natsbench.natsbench_regular_evaluator import NatsbenchRegularEvaluater
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
from archai.algos.naswotrain.naswotrain_natsbench_evaluator import NaswotrainNatsbenchEvaluater

from nats_bench import create
class NatsbenchRegularExperimentRunner(ExperimentRunner):
    """Runs freeze training on architectures from natsbench"""

    @overrides
    def model_desc_builder(self)->Optional[ModelDescBuilder]:
        return None

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
        evaler = NatsbenchRegularEvaluater()
        return evaler.evaluate(conf_eval, model_desc_builder=self.model_desc_builder())
        
