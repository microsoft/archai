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
from archai.algos.naswotrain.naswotrain_nasbench101_evaluater import NaswotrainNasbench101Evaluater
from archai.algos.nasbench101.nasbench101_dataset import Nasbench101Dataset
from .freeze_nasbench101_evaluater import FreezeNasbench101Evaluater

from nats_bench import create
class FreezeNasbench101ExperimentRunner(ExperimentRunner):
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
        naswotrain_evaler = NaswotrainNasbench101Evaluater()
        conf_eval_naswotrain = deepcopy(conf_eval)

        if conf_eval_naswotrain['checkpoint'] is not None:
            conf_eval_naswotrain['checkpoint']['filename'] = '$expdir/naswotrain_checkpoint.pth'

        naswotrain_eval_result = naswotrain_evaler.evaluate(conf_eval_naswotrain, model_desc_builder=self.model_desc_builder())
        logger.popd()

        # regular evaluation of the architecture
        # where we simply lookup the result
        # --------------------------------------
        logger.pushd('regular_evaluate')
        arch_id = conf_eval['nasbench101']['arch_index']
        dataroot = utils.full_path(conf_eval['loader']['dataset']['dataroot'])    
        # assuming that nasbench101 has been 'installed' in the dataroot folder
        nasbench101_location = os.path.join(dataroot, 'nasbench_ds', 'nasbench_only108.tfrecord.pkl')         
        dataset_name = conf_eval['loader']['dataset']['name']

        # create the nasbench101 api
        nsds = Nasbench101Dataset(nasbench101_location)

        # there are 423624 architectures total
        if arch_id < 0 or arch_id > 423623:
            logger.warn(f'architecture id {arch_id} is invalid ')
            raise NotImplementedError()

        if dataset_name != 'cifar10':
            logger.warn(f'dataset {dataset_name} is not part of nasbench101')
            raise NotImplementedError()

        data = nsds[arch_id]
        test_accuracy = data['avg_final_test_accuracy']

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
        conf_eval['model_filename'] = '$expdir/freeze_model.pt'
        
        if conf_eval['checkpoint'] is not None:
            conf_eval['checkpoint']['filename'] = '$expdir/freeze_checkpoint.pth'

        logger.pushd('freeze_evaluate')
        freeze_evaler = FreezeNasbench101Evaluater()
        conf_eval_freeze = deepcopy(conf_eval)
        freeze_eval_result = freeze_evaler.evaluate(conf_eval_freeze, model_desc_builder=self.model_desc_builder())
        logger.popd()

        return freeze_eval_result
