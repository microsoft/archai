from overrides import overrides
from typing import Optional, Type, Tuple

from archai.nas.exp_runner import ExperimentRunner
from archai.nas.model_desc_builder import ModelDescBuilder
from archai.nas.arch_trainer import TArchTrainer
from archai.common import common
from archai.common import utils
from archai.common.config import Config
from archai.nas.evaluater import Evaluater, EvalResult
from archai.nas.searcher import Searcher, SearchResult
from archai.nas.finalizers import Finalizers
from archai.nas.random_finalizers import RandomFinalizers
from archai.nas.model_desc_builder import ModelDescBuilder
from archai.algos.local_search_darts.local_search_darts_reg import LocalSearchDartsReg


class LocalSearchDartsRegExpRunner(ExperimentRunner):
    ''' Runs local search using regular evaluation on Natsbench space '''

    @overrides
    def trainer_class(self)->TArchTrainer:
        return None # no search trainer

    @overrides
    def run_search(self, conf_search:Config)->SearchResult:
        search = self.searcher()
        return search.search(conf_search)

    @overrides
    def searcher(self)->Searcher:
        return LocalSearchDartsReg()

    

