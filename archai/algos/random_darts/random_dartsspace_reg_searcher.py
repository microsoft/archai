import math as ma
from time import time
from typing import Set, Optional
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
from archai.algos.random.random_model_desc_builder import RandomModelDescBuilder


class RandomDartsSpaceRegSearcher(Searcher):

    @overrides
    def search(self, conf_search:Config, model_desc_builder:Optional[ModelDescBuilder],
                 trainer_class:TArchTrainer, finalizers:Finalizers)->SearchResult:

        assert type(model_desc_builder) == RandomModelDescBuilder, 'model_desc_builder must be RandomModelDescBuilder'

        # region config vars
        max_num_models = conf_search['max_num_models']
        dataroot = utils.full_path(conf_search['loader']['dataset']['dataroot'])
        dataset_name = conf_search['loader']['dataset']['name']
        conf_train = conf_search['trainer']
        conf_train_full = conf_search['trainer_full']
        conf_loader = conf_search['loader']
        # endregion

        best_trains = [(-1, -ma.inf)]
        models_used = dict()

        for i in range(max_num_models):
            # sample a model from darts search space
            # and evaluate it

            conf_model_desc = conf_search['model_desc']
            seed_for_arch_generation = random.randrange(10e18)
            # we don't load template model desc file from disk
            # as we are creating model based on seed
            model_desc = model_desc_builder.build(conf_model_desc, 
                                                seed=seed_for_arch_generation)
            model = self.model_from_desc(model_desc)
            

            checkpoint = None

            logger.pushd(f'regular_training_{i}')            
            data_loaders = self.get_data(conf_loader)
            trainer = Trainer(conf_train, model, checkpoint) 
            trainer_metrics = trainer.fit(data_loaders)
            train_time = trainer_metrics.total_training_time()
            logger.popd()

            this_arch_top1 = trainer_metrics.best_train_top1()    
            if this_arch_top1 > best_trains[-1][1]:
                best_trains.append((i, this_arch_top1, model.cpu()))
                
            # dump important things to log
            logger.pushd(f'best_trains_{i}')
            logger.info({'best_trains':best_trains})
            logger.popd()

        # Now take the model with best train and fully train it        
        best_trains.sort(key=lambda x:x[1], reverse=True)
        model_best = best_trains[0][2]
        logger.pushd('best_model_full_training')
        full_trainer = Trainer(conf_train_full, model_best, checkpoint)
        full_trainer.fit(data_loaders)
        logger.popd('best_model_full_training')














