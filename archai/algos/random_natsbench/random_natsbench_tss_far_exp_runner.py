

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
from archai.algos.random_natsbench.random_natsbench_tss_far_searcher import RandomNatsbenchTssFarSearcher


class RandomNatsbenchTssFarExpRunner(ExperimentRunner):
    ''' Runs random search using FastArchRank on Natsbench space '''

    @overrides
    def model_desc_builder(self)->Optional[ModelDescBuilder]:
        return None

    @overrides
    def trainer_class(self)->TArchTrainer:
        return None # no search trainer


    @overrides
    def run_search(self, conf_search:Config)->SearchResult:
        search = self.searcher()
        return search.search(conf_search)

    @overrides
    def run_eval(self, conf_eval:Config)->EvalResult:
        evaler = self.evaluater()
        return evaler.evaluate(conf_eval)

    @overrides
    def searcher(self)->Searcher:
        return RandomNatsbenchTssFarSearcher()

    @overrides
    def evaluater(self)->Evaluater:
        return None

    @overrides
    def copy_search_to_eval(self) -> None:
        return None
    

