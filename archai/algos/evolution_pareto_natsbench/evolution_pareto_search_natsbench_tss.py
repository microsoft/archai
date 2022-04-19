import os
from overrides.overrides import overrides
from typing import List, Tuple, Optional, Dict

import numpy as np

from archai.nas.discrete_search_space import DiscreteSearchSpace
from archai.nas.searcher import Searcher, SearchResult
from archai.common.common import logger
from archai.common.config import Config
from archai.common.trainer import Trainer
from archai.algos.evolution_pareto.evolution_pareto_search import EvolutionParetoSearch
from archai.nas.arch_meta import ArchWithMetaData
from archai.common import utils
from archai.search_spaces.discrete_search_spaces.natsbench_tss_search_spaces.discrete_search_space_natsbench_tss import DiscreteSearchSpaceNatsbenchTSS

from archai.nas.constraints.torch_constraints import measure_torch_inference_latency, measure_torch_peak_memory
from archai.nas.constraints.pareto_frontier import find_pareto_frontier_points

class EvolutionParetoSearchNatsbenchTSS(EvolutionParetoSearch):

    @overrides
    def search(self, conf_search:Config)->SearchResult:

        # region config vars
        self.dataroot = utils.full_path(conf_search['loader']['dataset']['dataroot'])
        self.dataset_name = conf_search['loader']['dataset']['name']
        self.natsbench_location = os.path.join(self.dataroot, 'natsbench', conf_search['natsbench']['natsbench_tss_fast'])
        self.conf_train = conf_search['trainer']
        self.conf_loader = conf_search['loader']
        # endregion

        # eval cache so that if search visits
        # a network already evaluated then we don't
        # evaluate it again. 
        self.eval_cache = {}

        super().search(conf_search)


    @overrides
    def get_search_space(self)->DiscreteSearchSpaceNatsbenchTSS:
        return DiscreteSearchSpaceNatsbenchTSS(self.dataset_name, 
                                               self.natsbench_location)



    @overrides
    def calc_memory_latency(self, population:List[ArchWithMetaData])->None:
        # computes memory and latency of each model
        # and updates the meta data
        for p in population:
            latency_s = measure_torch_inference_latency(p.arch, 
                                                        use_quantization=False, 
                                                        use_median=False,
                                                        input_dims=(1,3,32,32),
                                                        n_threads = 1,
                                                        n_trials=10,
                                                        device='cpu')

            peak_mem_mb = measure_torch_peak_memory(p.arch,
                                                    use_quantization=False,
                                                    input_dims=(1, 3, 32, 32),
                                                    n_threads=1,
                                                    device='cpu')

            p.metadata['latency'] = latency_s
            p.metadata['memory'] = peak_mem_mb



    @overrides
    def calc_task_accuracy(self, population:List[ArchWithMetaData])->None:
        # computes task accuracy of each model
        # and updates the meta data
        # TODO: parallelize it via ray in the future
        for p in population:
            train_top1 = self._evaluate(p) 
            p.metadata['train_top1'] = train_top1

        
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

        # # DEBUG: simulate architecture evaluation
        # train_top1 = random.random()
        # arch.metadata['train_top1'] = train_top1

        # cache it
        self.eval_cache[arch.metadata['archid']] = arch
        return train_top1


    @overrides
    def update_pareto_frontier(self, population:List[ArchWithMetaData])->List[ArchWithMetaData]:
        # need all decreasing or increasing quantities 
        all_errors = [1.0 - p.metadata['training_top1'] for p in population]
        all_latencies = [p.metadata['latency']  for p in population]
        all_memories = [p.metadata['memories']  for p in population]

        xs = np.array(all_errors)
        ys = np.array(all_latencies)
        zs = np.array(all_memories)
        
        points = np.concatenate((xs, ys, zs), axis=1)
        points_idx = find_pareto_frontier_points(points, is_decreasing=True)
        pareto_points = [population[idx] for idx in points_idx]
        return pareto_points


    @overrides
    def mutate_parents(self, parents:List[ArchWithMetaData])->List[ArchWithMetaData]:
        ''' Using the nearest neighbors as mutations'''
        mutations = []
        for p in parents:
            nbrs = self.search_space.get_neighbors(p)
            mutations.extend(nbrs)

        # TODO: there will be a lot of neighbors
        # so might want to downsample them
        return mutations


    @overrides
    def crossover_parents(self, parents:List[ArchWithMetaData])->List[ArchWithMetaData]:
        '''TODO: Returning empty for now '''
        return []

