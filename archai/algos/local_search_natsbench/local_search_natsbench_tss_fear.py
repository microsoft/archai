import os
from overrides.overrides import overrides
from typing import List, Tuple, Optional
import math as ma
from copy import deepcopy

from archai.nas.discrete_search_space import DiscreteSearchSpace
from archai.nas.searcher import Searcher, SearchResult
from archai.common.common import logger
from archai.common.config import Config
from archai.common.trainer import Trainer
from archai.algos.local_search.local_search import LocalSearch
from archai.nas.arch_meta import ArchWithMetaData
from archai.common import utils
from archai.algos.proxynas.conditional_trainer import ConditionalTrainer
from archai.algos.proxynas.freeze_trainer import FreezeTrainer
from archai.search_spaces.discrete_search_spaces.natsbench_tss_search_spaces.discrete_search_space_natsbench_tss import DiscreteSearchSpaceNatsbenchTSS


class LocalSearchNatsbenchTSSFear(LocalSearch):
    @overrides
    def search(self, conf_search:Config)->SearchResult:

        # region config vars
        self.max_num_models = conf_search['max_num_models']
        self.ratio_fastest_duration = conf_search['ratio_fastest_duration']
        self.dataroot = utils.full_path(conf_search['loader']['dataset']['dataroot'])
        self.dataset_name = conf_search['loader']['dataset']['name']
        self.natsbench_location = os.path.join(self.dataroot, 'natsbench', conf_search['natsbench']['natsbench_tss_fast'])
        self.conf_train = conf_search['trainer']
        self.conf_loader = conf_search['loader']
        self.conf_train_freeze = conf_search['freeze_trainer']
        # endregion

        # eval cache so that if local search visits
        # a network already evaluated then we don't
        # evaluate it again. 
        self.eval_cache = {}

        # cache of fear early rejects
        self.fear_early_rejects = {}

        # keep track of the fastest to train to 
        # threshold train/val accuracy
        self.fastest_cond_train = ma.inf

        super().search(conf_search)

    @overrides
    def get_search_space(self)->DiscreteSearchSpaceNatsbenchTSS:
        return DiscreteSearchSpaceNatsbenchTSS(self.dataset_name, 
                                                self.natsbench_location)


    @overrides
    def get_max_num_models(self)->int:
        return self.max_num_models


    @overrides
    def _check_membership(self, 
                        archs_touched: List[ArchWithMetaData], 
                        arch: ArchWithMetaData) -> bool:
        is_member = False
        for archmeta in archs_touched:
            if archmeta.metadata['archid'] == arch.metadata['archid']:
                is_member = True
        return is_member


    @overrides
    def _log_local_minima(self, curr_arch: ArchWithMetaData, 
                        curr_acc: float, 
                        num_evaluated: int) -> None:
        logger.pushd(f'local_minima_{num_evaluated}')
        curr_archid = curr_arch.metadata['archid']
        info = self.search_space.api.get_more_info(curr_archid, self.dataset_name, hp=200, is_random=False)
        curr_test_acc = info['test-accuracy']
        local_minimum = (curr_archid, curr_acc, curr_test_acc)
        logger.info({'output': local_minimum})
        self.local_minima.append(local_minimum)
        logger.popd()
        
        
    @overrides
    def _find_best_minimum(self)->Optional[Tuple[int, float, float]]:
        if self.local_minima:
            best_minimum = max(self.local_minima, key=lambda x:x[1])
            return best_minimum
        

    @overrides
    def _evaluate(self, arch:ArchWithMetaData)->float:
        
        # see if we have visited this arch before
        if arch.metadata['archid'] in self.eval_cache:
            logger.info(f"{arch.metadata['archid']} is in cache! Returning from cache.")
            return self.eval_cache[arch.metadata['archid']].metadata['train_top1']

        if arch.metadata['archid'] in self.fear_early_rejects:
            logger.info(f"{arch.metadata['archid']} has already been early rejected!")
            return
        
        # if not in cache actually evaluate it
        # -------------------------------------
        # NOTE: we don't pass checkpoint to the trainers
        # as it creates complications and we don't need it
        # as these trainers are quite fast
        checkpoint = None

        # if during conditional training it
        # starts exceeding fastest time to
        # reach threshold by a ratio then early
        # terminate it
        logger.pushd(f"conditional_training_{arch.metadata['archid']}")
        data_loaders = self.get_data(self.conf_loader)            
        time_allowed = self.ratio_fastest_duration * self.fastest_cond_train
        cond_trainer = ConditionalTrainer(self.conf_train, arch.arch, checkpoint, time_allowed) 
        cond_trainer_metrics = cond_trainer.fit(data_loaders)
        cond_train_time = cond_trainer_metrics.total_training_time()

        if cond_train_time >= time_allowed:
            # this arch exceeded time to reach threshold
            # cut losses and move to next one
            logger.info(f"{arch.metadata['archid']} exceeded time allowed. Terminating and ignoring.")
            self.fear_early_rejects[arch.metadata['archid']] = arch
            logger.popd()
            return

        if cond_train_time < self.fastest_cond_train:
            self.fastest_cond_train = cond_train_time
            logger.info(f'fastest condition train till now: {self.fastest_cond_train} seconds!')
        logger.popd()

        # if we did not early terminate in conditional 
        # training then freeze train
        # get data with new batch size for freeze training
        conf_loader_freeze = deepcopy(self.conf_loader)
        conf_loader_freeze['train_batch'] = self.conf_loader['freeze_loader']['train_batch'] 

        logger.pushd(f"freeze_training_{arch.metadata['archid']}")
        data_loaders = self.get_data(conf_loader_freeze, to_cache=False)
        # now just finetune the last few layers
        checkpoint = None
        trainer = FreezeTrainer(self.conf_train_freeze, arch.arch, checkpoint)
        freeze_train_metrics = trainer.fit(data_loaders)
        logger.popd()
        
        train_top1 = freeze_train_metrics.best_train_top1()
        arch.metadata['train_top1'] = train_top1
        # cache it
        self.eval_cache[arch.metadata['archid']] = arch
        return train_top1    




