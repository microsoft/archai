import math as ma
from typing import Set

from archai.nas.searcher import Searcher, SearchResult
from archai.common.config import Config
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
from archai.algos.natsbench.natsbench_utils import model_from_natsbench_tss

class RandomNatsbenchTssSearcher(Searcher):
    def search(conf_search:Config)->SearchResult:

        # region config vars
        max_num_models = conf_search['max_num_models']
        ratio_fastest_duration = conf_search['ratio_fastest_duration']
        top1_acc_threshold = conf_search['top1_acc_threshold']
        # endregion

        counter = 0
        best_trains = [-ma.Inf]
        fastest_cond_train = ma.Inf
        archids_sampled = set()
        while counter < max_num_models:
            # sample a random model from tss
            
            # if during conditional training it
            # starts exceeding fastest time to
            # reach threshold by a ratio then early
            # terminate it

            #  

        

        
