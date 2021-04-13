import math as ma
from typing import Set
import os
import random
from copy import deepcopy

from archai.nas.searcher import Searcher, SearchResult
from archai.common.config import Config
from archai.common.common import logger
from archai.nas.model_desc_builder import ModelDescBuilder
from archai.nas.arch_trainer import TArchTrainer
from archai.common.trainer import Trainer
from archai.nas.model_desc import CellType, ModelDesc
from archai.datasets import data
from archai.nas.model import Model
from archai.common.metrics import EpochMetrics, Metrics
from archai.common import utils
from archai.nas.finalizers import Finalizers
from archai.algos.proxynas.conditional_trainer import ConditionalTrainer
from archai.algos.proxynas.freeze_trainer import FreezeTrainer
from archai.algos.natsbench.natsbench_utils import create_natsbench_tss_api, model_from_natsbench_tss



class RandomNatsbenchTssSearcher(Searcher):
    def search(self, conf_search:Config)->SearchResult:

        # region config vars
        max_num_models = conf_search['max_num_models']
        ratio_fastest_duration = conf_search['ratio_fastest_duration']
        top1_acc_threshold = conf_search['top1_acc_threshold']
        dataroot = utils.full_path(conf_search['loader']['dataset']['dataroot'])
        dataset_name = conf_search['loader']['dataset']['name']
        natsbench_location = os.path.join(dataroot, 'natsbench', conf_search['natsbench']['natsbench_tss_fast'])
        conf_train = conf_search['trainer']
        conf_loader = conf_search['loader']
        conf_train_freeze = conf_search['freeze_trainer']
        # endregion

        # create the natsbench api
        api = create_natsbench_tss_api(natsbench_location, 'tss', fast_mode=True, verbose=True)

        # presample max number of archids without replacement
        random_archids = random.sample(range(api), k=max_num_models)

        best_trains = [(-1, -ma.Inf)]
        fastest_cond_train = ma.Inf
        
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
            logger.pushd('conditional training')
            data_loaders = self.get_data(conf_loader)
            time_allowed = ratio_fastest_duration * fastest_cond_train
            cond_trainer = ConditionalTrainer(conf_train, model, checkpoint, time_allowed) 
            cond_trainer_metrics = cond_trainer.fit(data_loaders)
            cond_train_time = cond_trainer_metrics.total_training_time()
            if  cond_train_time < fastest_cond_train:
                fastest_cond_train = cond_train_time
            logger.popd()

            # if we did not early terminate in conditional 
            # training then freeze train
            # get data with new batch size for freeze training
            # NOTE: important to create copy and modify as otherwise get_data will return
            # a cached data loader by hashing the id of conf_loader
            conf_loader_freeze = deepcopy(conf_loader)
            conf_loader_freeze['train_batch'] = conf_loader['freeze_loader']['train_batch'] 

            logger.pushd('freeze_training')
            data_loaders = self.get_data(conf_loader_freeze)
            # now just finetune the last few layers
            checkpoint = None
            trainer = FreezeTrainer(conf_train_freeze, model, checkpoint)
            freeze_train_metrics = trainer.fit(data_loaders)
            logger.popd()

            this_arch_top1 = freeze_train_metrics.best_train_top1()    
            if this_arch_top1 > best_trains[-1][1]:
                best_trains.append((archid, this_arch_top1))


        

        
