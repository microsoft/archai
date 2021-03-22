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
from archai.algos.zero_cost_measures.zero_cost_natsbench_evaluator import ZeroCostNatsbenchEvaluater

from nats_bench import create

class ZeroCostNatsbenchExperimentRunner(ExperimentRunner):
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
        # without training architecture evaluation score
        # ---------------------------------------
        logger.pushd('zerocost_evaluate')
        zerocost_evaler = ZeroCostNatsbenchEvaluater()
        conf_eval_zerocost = deepcopy(conf_eval)

        if conf_eval_zerocost['checkpoint'] is not None:
            conf_eval_zerocost['checkpoint']['filename'] = '$expdir/zerocost_checkpoint.pth'

        zerocost_eval_result = zerocost_evaler.evaluate(conf_eval_zerocost, model_desc_builder=self.model_desc_builder())
        logger.popd()

            
        # regular evaluation of the architecture
        # where we simply lookup the result
        # --------------------------------------
        dataset_name = conf_eval['loader']['dataset']['name']

        logger.pushd('regular_evaluate')   
        if dataset_name in {'cifar10', 'cifar100', 'ImageNet16-120'}:            
            arch_id = conf_eval['natsbench']['arch_index']
            dataroot = utils.full_path(conf_eval['loader']['dataset']['dataroot'])    
            natsbench_location = os.path.join(dataroot, 'natsbench', conf_eval['natsbench']['natsbench_tss_fast'])
            logger.info(natsbench_location)        
            
            api = create(natsbench_location, 'tss', fast_mode=True, verbose=True)
            
            if arch_id > 15625 or arch_id < 0:
                logger.warn(f'architecture id {arch_id} is invalid ')
                        
            info = api.get_more_info(arch_id, dataset_name, hp=200, is_random=False)
            test_accuracy = info['test-accuracy']
            logger.info(f'Regular training top1 test accuracy is {test_accuracy}')
            logger.info({'regtrainingtop1': float(test_accuracy)})
        else:
            logger.info({'regtrainingtop1': -1})
        logger.popd()
        

        return zerocost_eval_result