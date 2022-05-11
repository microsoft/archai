import os
from overrides.overrides import overrides
from typing import List, Tuple, Optional, Dict
import random

import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

from archai.common.common import get_conf
from archai.common.common import get_expdir
from archai.nas.discrete_search_space import DiscreteSearchSpace
from archai.nas.searcher import Searcher, SearchResult
from archai.common.common import logger
from archai.common.config import Config
from archai.common.trainer import Trainer
from archai.algos.bananas.bananas_search import BananasSearch
from archai.nas.arch_meta import ArchWithMetaData
from archai.nas.predictive_function import PredictiveFunction
from archai.common import utils
from archai.search_spaces.discrete_search_spaces.natsbench_tss_search_spaces.discrete_search_space_natsbench_tss import DiscreteSearchSpaceNatsbenchTSS

from archai.nas.constraints.torch_constraints import measure_torch_inference_latency, measure_torch_peak_memory
from archai.nas.constraints.pareto_frontier import find_pareto_frontier_points


class BananasSearchNatsbench(BananasSearch):

    @overrides
    def search(self, conf_search:Config)->SearchResult:

        # region config vars
        self.dataroot = utils.full_path(conf_search['loader']['dataset']['dataroot'])
        self.dataset_name = conf_search['loader']['dataset']['name']
        self.natsbench_location = os.path.join(self.dataroot, 'natsbench', conf_search['natsbench']['natsbench_tss_fast'])
        self.conf_train = conf_search['trainer']
        self.conf_loader = conf_search['loader']
        # if true uses table lookup to simulate training
        # else trains from scratch 
        self.use_benchmark = conf_search['use_benchmark']
        # if use_benchmark then evaluate 
        # architecture at this epoch
        self.evaluate_at_epoch = conf_search['evaluate_at_epoch']
        # number ensemble members to use for the 
        # predictions
        self.num_ensemble_members = conf_search['num_ensemble_members'] 
        # endregion


        # eval cache so that if search visits
        # a network already evaluated then don't
        # evaluate it again. 
        self.eval_cache = {}

        super().search(conf_search)


    @overrides
    def get_predictive_obj(self) -> PredictiveFunction:
        return PredictiveDNNEnsemble(self.num_ensemble_members)


    @overrides
    def get_search_space(self)->DiscreteSearchSpaceNatsbenchTSS:
        return DiscreteSearchSpaceNatsbenchTSS(self.dataset_name, 
                                               self.natsbench_location)


    @overrides
    def calc_task_accuracy(self, population:List[ArchWithMetaData])->None:
        # computes task accuracy of each model
        # and updates the meta data
        # TODO: parallelize it via ray in the future
        for p in tqdm(population):
            train_top1 = self._evaluate(p) 
            p.metadata['train_top1'] = train_top1


    def _evaluate(self, arch:ArchWithMetaData)->float:
        # since this is a tabular benchmark we
        # can potentially just return the value from 
        # the table at 'n' epochs
        if self.use_benchmark:
            # get training accuracy at 'n' epochs
            # from the benchmark
            train_top1 = self.search_space.get_training_accuracy_at_n_epoch(arch.metadata['archid'],
            datasetname=self.dataset_name,
            epoch=self.evaluate_at_epoch)
            return train_top1

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

        # # DEBUG: simulate architecture evaluation
        # train_top1 = random.random()
        # arch.metadata['train_top1'] = train_top1

        # cache it
        self.eval_cache[arch.metadata['archid']] = arch
        return train_top1

    @overrides
    def update_predictive_function(self) -> None:
        