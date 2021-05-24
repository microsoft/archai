import math as ma
from time import time
from typing import Set, List, Optional
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
from archai.nas.model_desc import CellType, ModelDesc
from archai.datasets import data
from archai.nas.model import Model
from archai.common.metrics import EpochMetrics, Metrics
from archai.common import utils
from archai.nas.finalizers import Finalizers
from archai.algos.proxynas.conditional_trainer import ConditionalTrainer
from archai.algos.proxynas.freeze_trainer import FreezeTrainer
from archai.algos.natsbench.natsbench_utils import create_natsbench_tss_api, model_from_natsbench_tss

class LocalNatsbenchTssFarSearcher(Searcher):

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

        # Natsbench Tss ops bag
        self.OPS = ['avg_pool_3x3', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'skip_connect']

        # create the natsbench api
        self.api = create_natsbench_tss_api(self.natsbench_location)

        # sanitize
        assert self.max_num_models <= len(self.api)
        assert self.ratio_fastest_duration >= 1.0

        # keep track of the fastest to train to 
        # threshold train/val accuracy
        self.fastest_cond_train = ma.inf

        # counter for models evaluated
        num_evaluated = 0

        # storage for archs touched till now
        archids_touched = []

        # sample an architecture at uniform random to 
        # initialize the search 
        prev_archid = -1
        prev_acc = -ma.inf
        curr_archid = random.sample(range(len(self.api)), k=1)

        # fear evaluate current archid
        curr_acc = self.fear_evaluate(curr_archid)
        archids_touched.append(curr_archid)
        num_evaluated += 1

        while curr_acc >= prev_acc:
            # get neighborhood of current model
            nbrhd_ids = self._get_neighbors(curr_archid)

            # evaluate neighborhood
            nbrhd_ids_accs = []
            for id in nbrhd_ids:
                if num_evaluated < self.max_num_models:
                    id_acc = self._fear_evaluate(id)
                    nbrhd_ids.append(id, id_acc)
                    archids_touched.append(id)
                    num_evaluated += 1
                else:
                    break

            # check if improved
            # NOTE: got to check that it can handle None values properly
            nbrhd_max_acc = max(nbrhd_ids_accs, key=lambda x: x[1])
            
            if nbrhd_max_acc >= curr_acc:
                prev_acc = curr_acc
                curr_acc = nbrhd_max_acc
                prev_archid = curr_archid
                curr_archid = [id_acc[1] for id_acc in nbrhd_ids_accs].index(nbrhd_max_acc)
            else:
                # didn't improve! this is a local minima
                # get the full evaluation result from natsbench
                info = self.api.get_more_info(curr_archid, self.dataset_name, hp=200, is_random=False)
                curr_test_acc = info['test-accuracy']
                logger.info({'output': (curr_archid, curr_acc, curr_test_acc)})
                # TODO: implement restarting search from another random point if budget 
                # is not exhausted
                if num_evaluated < self.max_num_models:
                    prev_archid = -1
                    prev_acc = -ma.inf
                    # sample an architecture not touched till now
                    curr_archid = None
                    while not curr_archid:
                        sampled_id = random.sample(range(len(self.api)), k=1)
                        if sampled_id not in archids_touched:
                            curr_archid = sampled_id

                    
                            
    def _get_neighbors(self, curr_archid:int)->List[int]:
        # first get the string representation of the current architecture
        config = self.api.get_net_config(curr_archid, self.dataset_name)
        string_rep = config.arch_str
        nbhd_strs = []
        ops = self._get_op_list(string_rep)
        for i in range(len(ops)):
            available = [op for op in self.OPS if op != ops[i]]
            for op in available:
                new_ops = ops.copy()
                new_ops[i] = op
                new_arch_str = {'string':self.get_string_from_ops(new_ops)}
                nbhd_strs.append(new_arch_str)

        # convert the arch strings to architecture ids
        # ind = self.api.archstr2index[config['arch_str']]
        nbhd_ids = []
        for arch_str in nbhd_strs:
            id = self.api.archstr2index[arch_str]
            nbhd_ids.append(id)
        return nbhd_ids
    
    def _get_op_list(self, string:str)->List[str]:
        # given a string, get the list of operations
        tokens = self.string.split('|')
        ops = [t.split('~')[0] for i,t in enumerate(tokens) if i not in [0,2,5,9]]
        return ops

    def _fear_evaluate(self, archid:int)->Optional[float]:
        model = model_from_natsbench_tss(archid, self.dataset_name, self.api)

        # NOTE: we don't pass checkpoint to the trainers
        # as it creates complications and we don't need it
        # as these trainers are quite fast
        checkpoint = None

        # if during conditional training it
        # starts exceeding fastest time to
        # reach threshold by a ratio then early
        # terminate it
        logger.pushd(f'conditional_training_{archid}')

        data_loaders = self.get_data(self.conf_loader)
        time_allowed = self.ratio_fastest_duration * self.fastest_cond_train
        cond_trainer = ConditionalTrainer(self.conf_train, model, checkpoint, time_allowed) 
        cond_trainer_metrics = cond_trainer.fit(data_loaders)
        cond_train_time = cond_trainer_metrics.total_training_time()

        if cond_train_time >= time_allowed:
            # this arch exceeded time to reach threshold
            # cut losses and move to next one
            logger.info(f'{archid} exceeded time allowed. Terminating and ignoring.')
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

        logger.pushd(f'freeze_training_{archid}')
        data_loaders = self.get_data(conf_loader_freeze, to_cache=False)
        # now just finetune the last few layers
        checkpoint = None
        trainer = FreezeTrainer(self.conf_train_freeze, model, checkpoint)
        freeze_train_metrics = trainer.fit(data_loaders)
        logger.popd()

        return freeze_train_metrics.best_train_top1()
            