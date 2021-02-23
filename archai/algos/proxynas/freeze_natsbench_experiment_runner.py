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
from archai.algos.naswotrain.naswotrain_natsbench_evaluator import NaswotrainNatsbenchEvaluater
from .freeze_natsbench_evaluater import FreezeNatsbenchEvaluater

from nats_bench import create
class FreezeNatsbenchExperimentRunner(ExperimentRunner):
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
        logger.pushd('naswotrain_evaluate')
        naswotrain_evaler = NaswotrainNatsbenchEvaluater()
        conf_eval_naswotrain = deepcopy(conf_eval)

        if conf_eval_naswotrain['checkpoint'] is not None:
            conf_eval_naswotrain['checkpoint']['filename'] = '$expdir/naswotrain_checkpoint.pth'

        naswotrain_eval_result = naswotrain_evaler.evaluate(conf_eval_naswotrain, model_desc_builder=self.model_desc_builder())
        logger.popd()

        # regular evaluation of the architecture
        # where we simply lookup the result
        # --------------------------------------
        logger.pushd('regular_evaluate')
        arch_id = conf_eval['natsbench']['arch_index']
        dataroot = utils.full_path(conf_eval['loader']['dataset']['dataroot'])    
        natsbench_location = os.path.join(dataroot, 'natsbench', conf_eval['natsbench']['natsbench_tss_fast'])
        logger.info(natsbench_location)        
        dataset_name = conf_eval['loader']['dataset']['name']

        api = create(natsbench_location, 'tss', fast_mode=True, verbose=True)
        
        if arch_id > 15625 or arch_id < 0:
            logger.warn(f'architecture id {arch_id} is invalid ')
        
        if dataset_name not in {'cifar10', 'cifar100', 'ImageNet16-120'}:
            logger.warn(f'dataset {dataset_name} is not part of natsbench')
            raise NotImplementedError()
        
        info = api.get_more_info(arch_id, dataset_name, hp=200, is_random=False)
        test_accuracy = info['test-accuracy']
        logger.info(f'Regular training top1 test accuracy is {test_accuracy}')
        logger.info({'regtrainingtop1': float(test_accuracy)})
        logger.popd()
            

        # freeze train evaluation of the architecture
        # -------------------------------------------
        # change relevant parts of conf_eval to ensure that freeze_evaler 
        # doesn't resume from checkpoints created by other evalers and saves 
        # models to different names as well
        conf_eval['full_desc_filename'] = '$expdir/freeze_full_model_desc.yaml'
        conf_eval['metric_filename'] = '$expdir/freeze_eval_train_metrics.yaml'
        conf_eval['model_filename'] = None # speed up download by not saving model
        
        if conf_eval['checkpoint'] is not None:
            conf_eval['checkpoint']['filename'] = '$expdir/freeze_checkpoint.pth'

        logger.pushd('freeze_evaluate')
        freeze_evaler = FreezeNatsbenchEvaluater()
        conf_eval_freeze = deepcopy(conf_eval)
        freeze_eval_result = freeze_evaler.evaluate(conf_eval_freeze, model_desc_builder=self.model_desc_builder())
        logger.popd()

        return freeze_eval_result
