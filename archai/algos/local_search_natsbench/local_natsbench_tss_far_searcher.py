import math as ma
from time import time
from typing import Set, List, Optional, Tuple
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
        self.use_fear = conf_search['use_fear']
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

        # store all local minima
        self.local_minima = []

        # cache fear evaluation results
        self.eval_cache = dict()

        # cache of fear early rejects
        self.fear_early_rejects = set()

        # counter for models evaluated
        num_evaluated = 0

        # storage for archs touched till now
        archids_touched = []

        # sample an architecture at uniform random to 
        # initialize the search 
        prev_acc = -ma.inf
        curr_archid = random.sample(range(len(self.api)), k=1)[0]

        # fear evaluate current archid
        curr_acc = self._evaluate_arch(curr_archid)
        archids_touched.append(curr_archid)
        num_evaluated += 1
        
        to_restart_search = False

        while curr_acc >= prev_acc:
            # get neighborhood of current model
            logger.info(f'current model {curr_archid}')
            nbrhd_ids = self._get_neighbors(curr_archid)

            # evaluate neighborhood
            nbrhd_ids_accs = []
            for id in nbrhd_ids:
                if num_evaluated < self.max_num_models:
                    id_acc = self._evaluate_arch(id)
                    archids_touched.append(id)
                    num_evaluated += 1
                    if id_acc:
                        nbrhd_ids_accs.append((id, id_acc))
                else:
                    break

            # check if improved
            if not nbrhd_ids_accs:
                logger.info('All neighbors got early rejected! FEAR thinks this is a local minima!')
                self._log_local_minima(curr_archid, curr_acc, num_evaluated)
                to_restart_search = True
            else:
                nbrhd_max_id_acc = max(nbrhd_ids_accs, key=lambda x: x[1])
                nbrhd_max_id = nbrhd_max_id_acc[0]
                nbrhd_max_acc = nbrhd_max_id_acc[1]
                if nbrhd_max_acc >= curr_acc:
                    # update current
                    prev_acc = curr_acc
                    curr_acc = nbrhd_max_acc
                    curr_archid = nbrhd_max_id
                    to_restart_search = False
                else:
                    # didn't improve! this is a local minima
                    # get the full evaluation result from natsbench
                    self._log_local_minima(curr_archid, curr_acc, num_evaluated)
                    to_restart_search = True

            # restarting search from another random point 
            # if budget is not exhausted
            if num_evaluated < self.max_num_models and to_restart_search:
                to_restart_search = False
                prev_acc = -ma.inf
                curr_archid = None
                curr_acc = None
                # sample an architecture not touched till now
                while not curr_archid or not curr_acc:
                    sampled_id = random.sample(range(len(self.api)), k=1)[0]
                    if sampled_id not in archids_touched:
                        curr_archid = sampled_id
                        # NOTE: fear evaluating could early reject!
                        curr_acc = self._evaluate_arch(curr_archid) 
                        archids_touched.append(curr_archid)
                        num_evaluated += 1
                        logger.info(f'restarting search with archid {curr_archid}')
            elif num_evaluated >= self.max_num_models:
                logger.info('terminating local search')
                best_minimum = self._find_best_minimum()
                logger.info({'best_minimum': best_minimum})
                logger.info({'all_minima': self.local_minima})
                break

        logger.info('terminating search outside while loop')

    def _log_local_minima(self, curr_archid:int, curr_acc:float, num_evaluated:int)->None:
        logger.pushd(f'local_minima_{num_evaluated}')
        info = self.api.get_more_info(curr_archid, self.dataset_name, hp=200, is_random=False)
        curr_test_acc = info['test-accuracy']
        local_minimum = (curr_archid, curr_acc, curr_test_acc)
        logger.info({'output': local_minimum})
        self.local_minima.append(local_minimum)
        logger.popd()

    def _find_best_minimum(self)->Tuple[int, float, float]:
        best_minimum = max(self.local_minima, key=lambda x:x[1])
        return best_minimum

                            
    def _get_neighbors(self, curr_archid:int)->List[int]:
        ''' Reused from https://github.com/naszilla/naszilla/blob/master/naszilla/nas_bench_201/cell_201.py '''
        # first get the string representation of the current architecture
        string_rep = self.api.get_net_config(curr_archid, self.dataset_name)['arch_str']
        nbhd_strs = []
        ops = self._get_op_list(string_rep)
        for i in range(len(ops)):
            available = [op for op in self.OPS if op != ops[i]]
            for op in available:
                new_ops = ops.copy()
                new_ops[i] = op
                new_arch_str = self._get_string_from_ops(new_ops)
                nbhd_strs.append(new_arch_str)

        # convert the arch strings to architecture ids
        nbhd_ids = []
        for arch_str in nbhd_strs:
            id = self.api.archstr2index[arch_str]
            nbhd_ids.append(id)
        return nbhd_ids

    
    def _get_op_list(self, string:str)->List[str]:
        ''' Reused from https://github.com/naszilla/naszilla/blob/master/naszilla/nas_bench_201/cell_201.py '''
        # given a string, get the list of operations
        tokens = string.split('|')
        ops = [t.split('~')[0] for i,t in enumerate(tokens) if i not in [0,2,5,9]]
        return ops


    def _reg_evaluate(self, archid:int)->float:

        # # DEBUG
        # # simulate regular evaluation
        # acc = random.random()
        # return acc

        # see if we have visited this architecture before
        if archid in self.eval_cache.keys():
            logger.info(f"{archid} is in cache! Returning from cache.")
            return self.eval_cache[archid]

        if archid in self.fear_early_rejects:
            logger.info(f"{archid} has already been early rejected!")
            return

        # if not in cache actually evaluate it
        model = model_from_natsbench_tss(archid, self.dataset_name, self.api)

        # NOTE: we don't pass checkpoint to the trainers
        # as it creates complications and we don't need it
        # as these trainers are quite fast
        checkpoint = None

        logger.pushd(f'regular_training_{archid}')            
        data_loaders = self.get_data(self.conf_loader)
        trainer = Trainer(self.conf_train, model, checkpoint) 
        trainer_metrics = trainer.fit(data_loaders)
        train_time = trainer_metrics.total_training_time()
        logger.popd()

        train_top1 = trainer_metrics.best_train_top1()
        # cache it
        self.eval_cache[archid] = train_top1
        return train_top1    


    def _fear_evaluate(self, archid:int)->Optional[float]:

        # # DEBUG
        # # simulate fear evaluation
        # acc = random.random()
        # if self.fastest_cond_train == ma.inf:
        #     self.fastest_cond_train = random.random() * 1000
        #     return acc
        # elif acc > 0.7:
        #     self.fastest_cond_train = random.random() * 1000
        #     return None
        # else:
        #     self.fastest_cond_train = random.random() * 1000
        #     return acc

        # see if we have visited this architecture before
        if archid in self.eval_cache.keys():
            logger.info(f"{archid} is in cache! Returning from cache.")
            return self.eval_cache[archid]

        if archid in self.fear_early_rejects:
            logger.info(f"{archid} has already been early rejected!")
            return
        
        # if not in cache actually evaluate it
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
            self.fear_early_rejects.add(archid)
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

        train_top1 = freeze_train_metrics.best_train_top1()
        # cache it
        self.eval_cache[archid] = train_top1
        return train_top1


    def _evaluate_arch(self, archid:int)->Optional[float]:
        if self.use_fear:
            curr_acc = self._fear_evaluate(archid)
        else:
            curr_acc = self._reg_evaluate(archid)
        return curr_acc

    
    def _get_string_from_ops(self, ops):
        ''' Reused from https://github.com/naszilla/naszilla/blob/master/naszilla/nas_bench_201/cell_201.py '''
        # given a list of operations, get the string
        strings = ['|']
        nodes = [0, 0, 1, 0, 1, 2]
        for i, op in enumerate(ops):
            strings.append(op+'~{}|'.format(nodes[i]))
            if i < len(nodes) - 1 and nodes[i+1] == 0:
                strings.append('+|')
        return ''.join(strings)