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
from archai.nas.model import Model
from archai.algos.proxynas.conditional_trainer import ConditionalTrainer
from archai.algos.proxynas.freeze_trainer import FreezeTrainer
from archai.algos.random.random_model_desc_builder import RandomModelDescBuilder


class RandomDartsSpaceFarSearcher(Searcher):

    @overrides
    def search(self, conf_search:Config, model_desc_builder:Optional[ModelDescBuilder],
                 trainer_class:TArchTrainer, finalizers:Finalizers)->SearchResult:

        assert type(model_desc_builder) == RandomModelDescBuilder, 'model_desc_builder must be RandomModelDescBuilder'

        # region config vars
        max_num_models = conf_search['max_num_models']
        ratio_fastest_duration = conf_search['ratio_fastest_duration']
        conf_train_freeze = conf_search['freeze_trainer']
        conf_train_full = conf_search['trainer_full']
        conf_loader = conf_search['loader']
        # endregion

        # NOTE: EXTREMELY important to get the seeds
        # beforehand as otherwise the seed for global
        # random module gets set by get_data calls
        # resulting in random's state being reset to the same
        # state and hence identical seeds being generated each call.
        seeds_for_arch_gen = []
        for i in range(max_num_models):
            seeds_for_arch_gen.append(random.randrange(10e18))

        best_trains = [(-1, -ma.inf)]
        fastest_cond_train = ma.inf

        for i in range(max_num_models):
            # sample a model from darts search space
            # and evaluate it

            conf_model_desc = conf_search['model_desc']
            # we don't load template model desc file from disk
            # as we are creating model based on seed
            model_desc = model_desc_builder.build(conf_model_desc, 
                                                seed=seeds_for_arch_gen[i])
            model = Model(model_desc, droppath=True, affine=True)

            logger.pushd(f'conditional_training_{i}')            
            data_loaders = self.get_data(conf_loader)
            time_allowed = ratio_fastest_duration * fastest_cond_train
            checkpoint = None
            cond_trainer = ConditionalTrainer(conf_train_full, model, checkpoint, time_allowed) 
            cond_trainer_metrics = cond_trainer.fit(data_loaders)
            cond_train_time = cond_trainer_metrics.total_training_time()

            if cond_train_time >= time_allowed:
                # this arch exceeded time to reach threshold
                # cut losses and move to next one
                logger.info(f'{i} exceeded time allowed. Terminating and ignoring.')
                logger.popd()
                continue

            if cond_train_time < fastest_cond_train:
                fastest_cond_train = cond_train_time
                logger.info(f'fastest condition train till now: {fastest_cond_train} seconds!')
            logger.popd()

            # if we did not early terminate in conditional 
            # training then freeze train
            # get data with new batch size for freeze training
            conf_loader_freeze = deepcopy(conf_loader)
            conf_loader_freeze['train_batch'] = conf_loader['freeze_loader']['train_batch'] 

            logger.pushd(f'freeze_training_{i}')
            data_loaders = self.get_data(conf_loader_freeze, to_cache=False)
            # now just finetune the last few layers
            checkpoint = None
            trainer = FreezeTrainer(conf_train_freeze, model, checkpoint)
            freeze_train_metrics = trainer.fit(data_loaders)
            logger.popd()

            this_arch_top1 = freeze_train_metrics.best_train_top1()    
            if this_arch_top1 > best_trains[-1][1]:
                best_trains.append((i, this_arch_top1, seeds_for_arch_gen[i]))

    
        # Now take the model with best train and fully train it        
        best_trains.sort(key=lambda x:x[1], reverse=True)
        model_seed_best = best_trains[0][2]
        model_desc_best = model_desc_builder.build(conf_model_desc, 
                                                seed=model_seed_best)
        model_best = Model(model_desc_best, droppath=True, affine=True)             
        logger.pushd('best_model_full_training')
        full_trainer = Trainer(conf_train_full, model_best, checkpoint)
        full_trainer.fit(data_loaders)
        logger.popd()
            