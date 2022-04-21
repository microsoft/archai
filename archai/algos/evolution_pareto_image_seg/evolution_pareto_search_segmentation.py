import os
from overrides.overrides import overrides
from typing import List, Tuple, Optional, Dict
import random

import torch

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
from archai.algos.evolution_pareto.evolution_pareto_search import EvolutionParetoSearch
from archai.nas.arch_meta import ArchWithMetaData
from archai.common import utils
from archai.search_spaces.discrete_search_spaces.segmentation_search_spaces.discrete_search_space_segmentation import DiscreteSearchSpaceSegmentation

from archai.algos.evolution_pareto_image_seg.segmentation_trainer import SegmentationTrainer

from archai.nas.constraints.torch_constraints import measure_torch_inference_latency, measure_torch_peak_memory
from archai.nas.constraints.pareto_frontier import find_pareto_frontier_points

class EvolutionParetoSearchSegmentation(EvolutionParetoSearch):

    @overrides
    def search(self, conf_search:Config)->SearchResult:

        # region config vars
        self.dataroot = utils.full_path(conf_search['loader']['dataset']['dataroot'])
        self.dataset_name = conf_search['loader']['dataset']['name']
        self.conf_train = conf_search['trainer']
        self.conf_loader = conf_search['loader']
        self.evaluate_at_epoch = conf_search['evaluate_at_epoch'] 
        # endregion

        # eval cache so that if search visits
        # a network already evaluated then we don't
        # evaluate it again. 
        self.eval_cache = dict()

        super().search(conf_search)


    @overrides
    def get_search_space(self)->DiscreteSearchSpaceSegmentation:
        return DiscreteSearchSpaceSegmentation(self.dataset_name)


    @overrides
    def calc_memory_latency(self, population:List[ArchWithMetaData])->None:
        # computes memory and latency of each model
        # and updates the meta data
        for p in tqdm(population):
            latency_s = measure_torch_inference_latency(p.arch, 
                                                        use_quantization=False, 
                                                        use_median=False,
                                                        input_dims=(1,3,32,32),
                                                        n_threads=1,
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
        # TODO: parallelize it via ray
        for p in tqdm(population):
            f1 = self._evaluate(p) 
            p.metadata['f1'] = f1


    def _evaluate(self, arch:ArchWithMetaData)->float:
        # see if we have visited this arch before
        if arch.metadata['archid'] in self.eval_cache:
            logger.info(f"{arch.metadata['archid']} is in cache! Returning from cache.")
            return self.eval_cache[arch.metadata['archid']].metadata['f1']
        
        # if not in cache actually evaluate it
        # -------------------------------------
        logger.pushd(f"regular_training_{arch.metadata['archid']}")

        # train
        dataset_dir = os.path.join(self.dataroot, 'face_synthetics')
        trainer = SegmentationTrainer(arch.arch, dataset_dir=self.dataroot, val_size=2000, gpus=1)
        trainer.fit(run_path=utils.full_path(get_expdir()))

        # validate
        val_dl = trainer.val_dataloader
        outputs = []

        with torch.no_grad():
            for bi, b in enumerate(tqdm(val_dl)):
                b['image'] = b['image'].to('cuda')
                b['mask'] = b['mask'].to('cuda')
                outputs.append(trainer.model.validation_step(b, bi))

        results = trainer.model.shared_epoch_end(outputs, stage='validation')

        logger.popd()

        train_top1 = results['validation_overall_f1']

        # # DEBUG: simulate architecture evaluation
        # train_top1 = random.random()
        # arch.metadata['train_top1'] = train_top1

        # cache it
        self.eval_cache[arch.metadata['archid']] = arch
        return train_top1