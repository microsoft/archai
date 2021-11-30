import os
import random
import math as ma
from overrides.overrides import overrides
from typing import List, Tuple, Optional, Dict
from archai.nas.discrete_search_space import DiscreteSearchSpace
from archai.nas.model_desc import ModelDesc

from archai.nas.searcher import Searcher, SearchResult
from archai.common.common import logger
from archai.common.config import Config
from archai.common.trainer import Trainer
from archai.algos.local_search.local_search import LocalSearch
from archai.nas.arch_meta import ArchWithMetaData
from archai.common import utils
from archai.search_spaces.discrete_search_spaces.darts_search_spaces.discrete_search_space_darts import DiscreteSearchSpaceDARTS


class LocalSearchDartsReg(LocalSearch):
    @overrides
    def search(self, conf_search:Config)->SearchResult:

        # region config vars
        self.max_num_models = conf_search['max_num_models']
        self.dataroot = utils.full_path(conf_search['loader']['dataset']['dataroot'])
        self.dataset_name = conf_search['loader']['dataset']['name']
        self.conf_train = conf_search['trainer']
        self.conf_loader = conf_search['loader']
        self.final_desc_filename = conf_search['final_desc_filename']
        # endregion

        # eval cache so that if local search visits
        # a network already evaluated then we don't
        # evaluate it again. 
        self.eval_cache:Dict[str, ArchWithMetaData] = {}

        super().search(conf_search)

        # save to the format that is expected by eval
        best_local_minimum = self._find_best_minimum()
        best_desc = best_local_minimum[0].arch.desc
        best_desc.save(self.final_desc_filename)



    @overrides
    def get_search_space(self)->DiscreteSearchSpaceDARTS:
        return DiscreteSearchSpaceDARTS()

    @overrides
    def get_max_num_models(self)->int:
        return self.max_num_models

    
    @overrides
    def _check_membership(self, 
                        archs_touched: List[ArchWithMetaData], 
                        arch: ArchWithMetaData) -> bool:
        is_member = False
        for archmeta in archs_touched:
            if archmeta.arch.desc == arch.arch.desc:
                is_member = True
        return is_member


    @overrides
    def _log_local_minima(self, curr_arch: ArchWithMetaData, 
                        curr_acc: float, 
                        num_evaluated: int) -> None:
        logger.pushd(f'local_minima_{num_evaluated}')
        local_minimum = (curr_arch, curr_acc)
        logger.info({'output': local_minimum})
        self.local_minima.append(local_minimum)
        logger.popd()
        
        
    @overrides
    def _find_best_minimum(self)->Optional[Tuple[ArchWithMetaData, float]]:
        if self.local_minima:
            best_minimum = max(self.local_minima, key=lambda x:x[1])
            return best_minimum
        else:
            # if no local minima encountered, return the best 
            # encountered so far
            logger.info('No local minima encountered! Returning best of visited.')
            max_train_top1 = -ma.inf
            argmax_arch = None
            for _, arch in self.eval_cache.items():
                this_train_top1 = arch.metadata['train_top1']
                if this_train_top1 > max_train_top1:
                    max_train_top1 = this_train_top1
                    argmax_arch = arch
            return argmax_arch, max_train_top1
            


    @overrides
    def _evaluate(self, arch:ArchWithMetaData)->float:
        arch_flat_rep = arch.arch.desc.get_flat_rep()

        # see if we have visited this arch before
        if arch_flat_rep in self.eval_cache:
            logger.info(f"Arch is in eval cache. Returning from cache.")
            return self.eval_cache[arch_flat_rep].metadata['train_top1']

        # if not in cache actually evaluate it
        # -------------------------------------
        # NOTE: we don't pass checkpoint to the trainers
        # as it creates complications and we don't need it
        # as these trainers are quite fast
        checkpoint = None

        logger.pushd(f"regular_training_{arch.metadata['archid']}")            
        data_loaders = self.get_data(self.conf_loader)
        trainer = Trainer(self.conf_train, arch.arch, checkpoint) 
        trainer_metrics = trainer.fit(data_loaders)
        train_time = trainer_metrics.total_training_time()
        logger.popd()

        train_top1 = trainer_metrics.best_train_top1()
        arch.metadata['train_top1'] = train_top1

        # # DEBUG: simulate architecture evaluation
        # train_top1 = random.random()
        # arch.metadata['train_top1'] = train_top1

        # cache it
        self.eval_cache[arch_flat_rep] = arch
        return train_top1    