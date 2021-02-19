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
from archai.algos.nasbench101.nasbench101_dataset import Nasbench101Dataset
from archai.algos.nasbench101.nasbench101_evaluater import Nb101RegularEvaluater


class Nb101RegularExperimentRunner(ExperimentRunner):
    """Runs regular training on architectures from nasbench101"""

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

        # regular evaluation of the architecture
        # where we simply lookup the result
        # --------------------------------------
        logger.pushd('regular_evaluate')
        arch_id = conf_eval['nasbench101']['arch_index']
        dataroot = utils.full_path(conf_eval['loader']['dataset']['dataroot'])    
        # assuming that nasbench101 has been 'installed' in the dataroot folder
        nasbench101_location = os.path.join(dataroot, 'nasbench_ds', 'nasbench_full.pkl')         
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

        all_trials = nsds.get_test_acc(arch_id)
        assert len(all_trials) > 0
        test_accuracy = sum(all_trials) / len(all_trials)

        logger.info(f'Regular training top1 test accuracy is {test_accuracy}')
        logger.info({'regtrainingtop1': float(test_accuracy)})
        logger.popd()
        
        # regular evaluation of n epochs
        evaler = Nb101RegularEvaluater()
        return evaler.evaluate(conf_eval, model_desc_builder=self.model_desc_builder())
        
