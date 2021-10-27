from abc import ABCMeta, abstractmethod
import math as ma
from typing import Tuple, List

import torch.nn as nn

from archai.common.common import logger
from archai.nas.arch_meta import ArchWithMetaData
from archai.nas.discrete_search_space import DiscreteSearchSpace


class LocalSearch(metaclass=ABCMeta):
    def __init__(self, max_num_models:int, search_space:DiscreteSearchSpace):
        assert max_num_models >= 0
        self.max_num_models = max_num_models
        assert isinstance(search_space, DiscreteSearchSpace)
        self.search_space = search_space

        # store all local minima
        self.local_minima = []


    def search(self):
        num_evaluated = 0
        archs_touched = []

        # sample an architecture at uniform random
        # to initialize the search
        prev_acc = -ma.inf
        curr_arch = self.search_space.random_sample()
        curr_acc = self._evaluate(curr_arch)
        archs_touched.append(curr_arch)
        num_evaluated += 1

        to_restart_search = False
        while curr_acc >= prev_acc:
            # get neighborhood of current model
            logger.info(f'current_model {curr_arch}')
            nbrhd_archs = self.search_space.get_neighbors(curr_arch)

            # evaluate neighborhood
            nbrhd_archs_accs = []
            for arch in nbrhd_archs:
                if num_evaluated < self.max_num_models:
                    acc = self._evaluate(arch)
                    archs_touched.append(arch)
                    num_evaluated += 1
                    if acc:
                        nbrhd_archs_accs.append((arch, acc))
                else:
                    break

            # check if improved
            if not nbrhd_archs_accs:
                logger.info('All neighbors got early rejected!')
                self._log_local_minima(curr_arch, curr_acc, num_evaluated)
                to_restart_search = True
            else:
                nbrhd_max_arch_acc = max(nbrhd_archs_accs, key=lambda x: x[1])
                nbrhd_max_arch = nbrhd_max_arch_acc[0]
                nbrhd_max_acc = nbrhd_max_arch_acc[1]
                if nbrhd_max_acc >= curr_acc:
                    # update current
                    prev_acc = curr_acc
                    curr_acc = nbrhd_max_acc
                    curr_arch = nbrhd_max_arch
                    to_restart_search = False
                else:
                    # didn't improve! this is a local minima
                    # get the full evaluation result from natsbench
                    self._log_local_minima(curr_arch, curr_acc, num_evaluated)
                    to_restart_search = True

            # restarting search from another random point 
            # if budget is not exhausted
            if num_evaluated < self.max_num_models and to_restart_search:
                to_restart_search = False
                prev_acc = -ma.inf
                curr_acc = None
                # sample an architecture not touched till now
                while not curr_arch or not curr_acc:
                    sampled_arch = self.search_space.random_sample()
                    if not self._check_membership(archs_touched, sampled_arch)
                        curr_arch = sampled_arch
                        # NOTE: some evaluation method could early reject!
                        curr_acc = self._evaluate(curr_arch) 
                        archs_touched.append(curr_arch)
                        num_evaluated += 1
                        logger.info(f'restarting search with arch {curr_arch}')
            elif num_evaluated >= self.max_num_models:
                logger.info('terminating local search')
                best_minimum = self._find_best_minimum()
                logger.info({'best_minimum': best_minimum})
                logger.info({'all_minima': self.local_minima})
                break


    @abstractmethod
    def _evaluate(self, arch:ArchWithMetaData)->float:
        pass

    @abstractmethod
    def _check_membership(self, 
                        archs_touched:List[ArchWithMetaData],
                        archs:ArchWithMetaData)->bool:
        pass

    @abstractmethod
    def _log_local_minima(self, 
                        curr_arch:ArchWithMetaData, 
                        curr_acc:float, 
                        num_evaluated:int)->None:
        pass

    @abstractmethod
    def _find_best_minimum(self)->Tuple[int, float, float]:
        pass
    

































