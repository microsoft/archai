import os
from overrides.overrides import overrides
from typing import List, Tuple, Optional, Dict
import random
import yaml
import ray

import torch

import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

from archai.common.common import get_conf, get_conf_common
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

from archai.algos.evolution_pareto_image_seg.segmentation_trainer import SegmentationTrainer, get_custom_overall_metrics
from archai.algos.evolution_pareto_image_seg.utils import profile_torch, profile_onnx, get_onnx_latency, to_onnx

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
        self.min_mac = conf_search['min_mac']
        self.max_mac = conf_search['max_mac']
        self.min_layers = conf_search['min_layers']
        self.max_layers = conf_search['max_layers']
        self.max_downsample_factor = conf_search['max_downsample_factor']
        self.skip_connections = conf_search['skip_connections']
        self.max_skip_connection_length = conf_search['max_skip_connection_length']
        self.max_scale_delta = conf_search['max_scale_delta']
        # endregion

        # eval cache so that if search visits
        # a network already evaluated then we don't
        # evaluate it again. 
        self.eval_cache = dict()

        # init ray
        ray.init()

        super().search(conf_search)


    @overrides
    def get_search_space(self)->DiscreteSearchSpaceSegmentation:
        return DiscreteSearchSpaceSegmentation(self.dataset_name, 
                                               min_mac=self.min_mac, 
                                               max_mac=self.max_mac)


    @overrides
    def calc_memory_latency(self, population:List[ArchWithMetaData])->None:
        # computes memory and latency of each model
        # and updates the meta data
        
        for p in tqdm(population):
            
            # TODO: verify that the unit is ms
            latency_ms = get_onnx_latency(p.arch, img_size=p.arch.img_size)

            # TODO: get peak memory of the onnx model
            # instead of the torch model
            peak_mem_mb = measure_torch_peak_memory(p.arch,
                                                    use_quantization=False,
                                                    input_dims=(1, 3, p.arch.img_size, p.arch.img_size),
                                                    n_threads=1,
                                                    device='cpu')

            p.metadata['latency'] = latency_ms
            p.metadata['memory'] = peak_mem_mb


    @overrides
    def calc_task_accuracy(self, population:List[ArchWithMetaData])->None:
        # computes task accuracy of each model
        # and updates the meta data
        # TODO: parallelize it via ray

        # folder where to store training logs of each model
        exp_dir = utils.full_path(get_expdir())
        save_folder = os.path.join(exp_dir, f'arch_eval_logs_iter_{self.iter_num}')
        os.makedirs(save_folder, exist_ok=True)

        fit_refs = []
        for p in population:
            # create a ray actor per model to be trained
            actor_ref = self._create_training_job(p)
            # create a folder name for the model training logs
            run_path = os.path.join(save_folder, str(p.metadata['archid']))
            os.makedirs(run_path, exist_ok=True)
            # fit and validate the model 
            fit_refs.append(actor_ref.fit_and_validate.remote(run_path=run_path))
        
        # gather all results for all models
        results = ray.get(fit_refs) # TODO: make sure these are ordered otherwise bug

        for r, p in zip(results, population):
            p.metadata['f1'] = r

    def _create_training_job(self, arch:ArchWithMetaData)->List:
        ''' Creates a ray actor that will train a single architecture '''
        # region config
        self.evaluate_for_steps = self.conf_train['evaluate_for_steps']  
        self.val_size = self.conf_train['val_size']
        self.gpus = self.conf_train['gpus']
        self.img_size = self.conf_train['img_size']
        self.augmentation = self.conf_train['augmentation']
        self.lr = self.conf_train['lr']
        self.criterion_name = self.conf_train['criterion_name']
        self.batch_size = self.conf_loader['batch_size']
        self.seed = get_conf_common()['seed']
        # region

        # train
        dataset_dir = os.path.join(self.dataroot, 'face_synthetics')
        ref = SegmentationTrainer.remote(arch.arch, 
                                      dataset_dir=dataset_dir, 
                                      max_steps=self.evaluate_for_steps,
                                      val_size=self.val_size,
                                      img_size=self.img_size,
                                      augmentation=self.augmentation,
                                      batch_size=self.batch_size,
                                      lr=self.lr,
                                      criterion_name=self.criterion_name, 
                                      gpus=self.gpus,
                                      seed=self.seed)
        return ref

    @overrides
    def update_pareto_frontier(self, population:List[ArchWithMetaData])->List[ArchWithMetaData]:
        # need all decreasing or increasing quantities 
        all_errors = [1.0 - p.metadata['f1'] for p in population]
        all_latencies = [p.metadata['latency']  for p in population]
        all_memories = [p.metadata['memory']  for p in population]

        xs = np.array(all_errors).reshape(-1, 1)
        ys = np.array(all_latencies).reshape(-1, 1)
        zs = np.array(all_memories).reshape(-1, 1)
        
        points = np.concatenate((xs, ys, zs), axis=1)
        points_idx = find_pareto_frontier_points(points, is_decreasing=True)
        pareto_points = [population[idx] for idx in points_idx]

        # save all the pareto points
        self._save_yaml(pareto_points, basename='pareto')

        return pareto_points


    def _save_yaml(self, points:List[ArchWithMetaData], basename='pareto')->None:
        exp_dir = utils.full_path(get_expdir())
        save_folder = os.path.join(exp_dir, f'{basename}_iter_{self.iter_num}')
        os.makedirs(save_folder, exist_ok=True)
        for p in points:
            this_name = os.path.join(save_folder, str(p.metadata['archid']) + '.yaml')
            save_dict = {
                            'channels_per_scale': p.metadata['channels_per_scale'],
                            'architecture': p.metadata['graph']
                        }
            with open(this_name, 'w') as f:
                _ = yaml.dump(save_dict, f)
        


    @overrides
    def mutate_parents(self, parents:List[ArchWithMetaData])->List[ArchWithMetaData]:
        ''' Using the nearest neighbors as mutations'''
        mutations = []
        for p in parents:
            # TODO: this only returns one neighbor
            # may want to sample more
            nbrs = self.search_space.get_neighbors(p)
            mutations.extend(nbrs)

        # TODO: there will be a lot of neighbors
        # so might want to downsample them
        return mutations


    @overrides
    def crossover_parents(self, parents:List[ArchWithMetaData])->List[ArchWithMetaData]:
        '''TODO: Returning empty for now '''
        return []


    @overrides
    def plot_search_state(self, all_pop:List[ArchWithMetaData], pareto:List[ArchWithMetaData], iter_num:int) -> None:
        all_accs = [p.metadata['f1'] for p in all_pop]
        all_latencies = [p.metadata['latency']  for p in all_pop]
        all_memories = [p.metadata['memory']  for p in all_pop]

        p_accs = [p.metadata['f1'] for p in pareto]
        p_latencies = [p.metadata['latency']  for p in pareto]
        p_memories = [p.metadata['memory']  for p in pareto]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=all_accs, 
                                y=all_latencies, 
                                z=all_memories,
                                mode='markers',
                                marker_color='blue',
                                showlegend=True,
                                name='All visited architectures'))

        fig.add_trace(go.Scatter3d(x=p_accs, 
                                y=p_latencies, 
                                z=p_memories,
                                mode='markers',
                                marker_color='red',
                                showlegend=True,
                                name='Pareto architectures'))
        
        title_text = f'Search State Iter {iter_num}'
        xaxis_title = 'Accuracy (validation f1)'
        yaxis_title = 'Latency (ms)'
        zaxis_title = 'Memory (mb)'
        fig.update_layout(title_text=title_text,
                          scene=dict(xaxis_title=xaxis_title,
                                     yaxis_title=yaxis_title,
                                     zaxis_title=zaxis_title))

        expdir = get_expdir()
        html_path = os.path.join(expdir, f'search_state_{iter_num}.html')
        fig.write_html(html_path)

        png_path = os.path.join(expdir, f'search_state_{iter_num}.png')
        fig.write_image(png_path, engine='kaleido', width=1500, height=1500, scale=1) 
