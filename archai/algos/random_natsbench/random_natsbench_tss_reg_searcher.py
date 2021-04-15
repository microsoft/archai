import math as ma
from time import time
from typing import Set
import os
import random
from copy import deepcopy

from overrides import overrides

from archai.nas.searcher import Searcher, SearchResult
from archai.common.config import Config
from archai.common.common import logger
from archai.nas.model_desc_builder import ModelDescBuilder
from archai.nas.arch_trainer import TArchTrainer
from archai.common.trainer import Trainer
from archai.common import utils
from archai.nas.finalizers import Finalizers
from archai.algos.proxynas.conditional_trainer import ConditionalTrainer
from archai.algos.proxynas.freeze_trainer import FreezeTrainer
from archai.algos.natsbench.natsbench_utils import create_natsbench_tss_api, model_from_natsbench_tss



class RandomNatsbenchTssRegSearcher(Searcher):

    @overrides
    def search(self, conf_search:Config)->SearchResult:

        # region config vars
        max_num_models = conf_search['max_num_models']
        dataroot = utils.full_path(conf_search['loader']['dataset']['dataroot'])
        dataset_name = conf_search['loader']['dataset']['name']
        natsbench_location = os.path.join(dataroot, 'natsbench', conf_search['natsbench']['natsbench_tss_fast'])
        conf_train = conf_search['trainer']
        conf_loader = conf_search['loader']
        # endregion

        # create the natsbench api
        api = create_natsbench_tss_api(natsbench_location)

        # presample max number of archids without replacement
        random_archids = random.sample(range(len(api)), k=max_num_models)

        best_trains = [(-1, -ma.inf)]
        best_tests = [(-1, -ma.inf)]
        
        for archid in random_archids:
            # get model
            model = model_from_natsbench_tss(archid, dataset_name, api)

            # NOTE: we don't pass checkpoint to the trainers
            # as it creates complications and we don't need it
            # as these trainers are quite fast
            checkpoint = None

            # if during conditional training it
            # starts exceeding fastest time to
            # reach threshold by a ratio then early
            # terminate it
            logger.pushd(f'regular_training_{archid}')
            
            data_loaders = self.get_data(conf_loader)
            trainer = Trainer(conf_train, model, checkpoint) 
            trainer_metrics = trainer.fit(data_loaders)
            train_time = trainer_metrics.total_training_time()
            logger.popd()

            this_arch_top1 = trainer_metrics.best_train_top1()    
            if this_arch_top1 > best_trains[-1][1]:
                best_trains.append((archid, this_arch_top1))

                # get the full evaluation result from natsbench
                info = api.get_more_info(archid, dataset_name, hp=200, is_random=False)
                this_arch_top1_test = info['test-accuracy']
                best_tests.append((archid, this_arch_top1_test))

            # dump important things to log
            logger.pushd(f'best_trains_tests_{archid}')
            logger.info({'best_trains':best_trains, 'best_tests':best_tests})
            logger.popd()

