import os
from overrides.overrides import overrides
from typing import List, Tuple
from archai.nas.discrete_search_space import DiscreteSearchSpace

from archai.nas.searcher import Searcher, SearchResult
from archai.common.common import logger
from archai.common.config import Config
from archai.common.trainer import Trainer
from archai.algos.local_search.local_search import LocalSearch
from archai.nas.arch_meta import ArchWithMetaData
from archai.common import utils
from archai.search_spaces.discrete_search_spaces.natsbench_tss_search_spaces.discrete_search_space_natsbench_tss import DiscreteSearchSpaceNatsbenchTSS


class LocalSearchNatsbenchTSSReg(LocalSearch):
    @overrides
    def search(self, conf_search:Config)->SearchResult:

        # region config vars
        self.max_num_models = conf_search['max_num_models']
        self.dataroot = utils.full_path(conf_search['loader']['dataset']['dataroot'])
        self.dataset_name = conf_search['loader']['dataset']['name']
        self.natsbench_location = os.path.join(self.dataroot, 'natsbench', conf_search['natsbench']['natsbench_tss_fast'])
        self.conf_train = conf_search['trainer']
        self.conf_loader = conf_search['loader']
        # endregion

        # eval cache so that if local search visits
        # a network already evaluated then we don't
        # evaluate it again. 
        self.eval_cache = {}

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
    def _find_best_minimum(self)->Tuple[int, float, float]:
        best_minimum = max(self.local_minima, key=lambda x:x[1])
        return best_minimum
        

    @overrides
    def _evaluate(self, arch:ArchWithMetaData)->float:
        
        # see if we have visited this arch before
        if arch.metadata['archid'] in self.eval_cache:
            logger.info(f"{arch.metadata['archid']} is in cache! Returning from cache.")
            return self.eval_cache[arch.metadata['archid']].metadata['train_top1']

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
        # cache it
        self.eval_cache[arch.metadata['archid']] = arch
        return train_top1    




