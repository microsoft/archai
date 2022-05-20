import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from overrides.overrides import overrides
from typing import List, Tuple, Optional, Dict
import pandas as pd
import random
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
from archai.nas.nas_utils import compute_crowding_distance, compute_pareto_hypervolume
from archai.common import utils
from archai.search_spaces.discrete_search_spaces.segmentation_search_spaces.discrete_search_space_segmentation import DiscreteSearchSpaceSegmentation

from archai.algos.evolution_pareto_image_seg.segmentation_trainer import SegmentationTrainer, get_custom_overall_metrics
from archai.algos.evolution_pareto_image_seg.utils import get_onnx_latency, to_onnx, get_utc_date
from archai.algos.evolution_pareto_image_seg.report import get_search_status_df, save_3d_pareto_plot, save_2d_pareto_evolution_plot
from archai.algos.evolution_pareto_image_seg.remote_benchmark import RemoteAzureBenchmark

from archai.nas.constraints.torch_constraints import measure_torch_inference_latency, measure_torch_peak_memory
from archai.nas.constraints.pareto_frontier import find_pareto_frontier_points

class EvolutionParetoSearchSegmentation(EvolutionParetoSearch):

    @overrides
    def search(self, conf_search:Config)->SearchResult:

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
        self.max_post_upsample_layers = conf_search['max_post_upsample_layers']
        self.min_base_channels = conf_search['min_base_channels']
        self.max_base_channels = conf_search['max_base_channels']
        self.base_channels_binwidth = conf_search['base_channels_binwidth']
        self.min_delta_channels = conf_search['min_delta_channels']
        self.max_delta_channels = conf_search['max_delta_channels']
        self.delta_channels_binwidth = conf_search['delta_channels_binwidth']
        self.op_subset = conf_search['op_subset']
        self.downsample_prob_ratio = conf_search['downsample_prob_ratio']

        self.objectives = conf_search['objectives']

        self.crowd_sorting = conf_search['crowd_sorting']

        self.init_architectures_from_dir = conf_search['init_architectures_from_dir']
        self.use_remote_benchmark = conf_search['use_remote_benchmark']

        if self.use_remote_benchmark:
            remote_config = conf_search['remote_benchmark_config']
            assert 'connection_string_env_var_name' in remote_config
            assert remote_config['connection_string_env_var_name'] in os.environ

            con_string = os.environ[remote_config['connection_string_env_var_name']]
            self.patience = remote_config['patience']
            self.check_interval = remote_config['check_interval']

            self.remote_benchmark = RemoteAzureBenchmark(
                connection_string=con_string, 
                blob_container_name=remote_config['blob_container_name'],
                table_name=remote_config['table_name'],
                partition_key=remote_config['partition_key'],
                overwrite=remote_config['overwrite']
            )

        # eval cache so that if search visits
        # a network already evaluated then we don't
        # evaluate it again. 
        self.eval_cache = dict()

        # Place to store models with evaluation errors
        self.models_with_missing_results = []

        # init ray
        ray.init()

        super().search(conf_search)


    @overrides
    def get_search_space(self)->DiscreteSearchSpaceSegmentation:
        return DiscreteSearchSpaceSegmentation(
            self.dataset_name,
            min_layers=self.min_layers,
            max_layers=self.max_layers,
            max_downsample_factor=self.max_downsample_factor,
            skip_connections=self.skip_connections,
            max_skip_connection_length=self.max_skip_connection_length,
            max_scale_delta=self.max_scale_delta,
            min_base_channels=self.min_base_channels,
            max_base_channels=self.max_base_channels,
            base_channels_binwidth=self.base_channels_binwidth,
            min_delta_channels=self.min_delta_channels,
            max_delta_channels=self.max_delta_channels,
            delta_channels_binwidth=self.delta_channels_binwidth,
            min_mac=self.min_mac, 
            max_mac=self.max_mac,
            op_subset=self.op_subset,
            downsample_prob_ratio=self.downsample_prob_ratio
        )

    def _get_proxy_memory_latency(self, model: ArchWithMetaData) -> Tuple[float, float]:
        memory, latency = 0, 0
        
        if self.objectives['latency']['enabled']:
            latency = get_onnx_latency(model.arch, img_size=model.arch.img_size)

        if self.objectives['memory']['enabled']:
            memory = measure_torch_peak_memory(
                model.arch, use_quantization=False,
                input_dims=(1, 3, model.arch.img_size, model.arch.img_size), 
                n_threads=1, device='cpu'
            )

        return memory, latency

    @overrides
    def _sample_init_population(self) -> List[ArchWithMetaData]:
        # Manual initialization
        if self.init_architectures_from_dir:
            arch_dir = Path(self.init_architectures_from_dir)
            arch_files = list(arch_dir.glob('*.yaml'))
            search_space = self.get_search_space()
            logger.info(f'Loading {len(arch_files)} seed models for first iteration.')

            return [search_space.load_from_file(arch_file) for arch_file in arch_files]
        
        # Initialization with crowd sorting
        if self.crowd_sorting['initialization']:
            init_pop = []

            for _ in range(self.crowd_sorting['oversampling_factor']):
                init_pop += super()._sample_init_population()

            # Scores memory and latency
            proxy_mem_latency = np.array([
                list(self._get_proxy_memory_latency(p)) for p in init_pop
            ])

            crowd_dist = compute_crowding_distance(proxy_mem_latency)
            idxs = np.argsort(-crowd_dist, axis=None)[:self.init_num_models]
            model_list = [p for pi, p in enumerate(init_pop) if pi in idxs]
        else:
            model_list = super()._sample_init_population()

        for model in model_list:
            model.metadata['generation'] = self.iter_num

        return model_list

    @overrides
    def _sample_random_to_mix(self) -> List[ArchWithMetaData]:
        if self.crowd_sorting['random_mix']:
            init_pop = []

            for _ in range(self.crowd_sorting['oversampling_factor']):
                init_pop += super()._sample_random_to_mix()

            # Scores memory and latency
            proxy_mem_latency = np.array([
                list(self._get_proxy_memory_latency(p)) for p in init_pop
            ])

            crowd_dist = compute_crowding_distance(proxy_mem_latency)
            idxs = np.argsort(-crowd_dist, axis=None)[:self.num_random_mix]
            model_list = [p for pi, p in enumerate(init_pop) if pi in idxs]
        else:
            model_list = super()._sample_random_to_mix()

        for model in model_list:
            model.metadata['generation'] = self.iter_num

        return model_list

    @overrides
    def calc_memory_latency(self, population:List[ArchWithMetaData])->None:
        # computes memory and latency of each model
        cache_misses = 0
        for p in tqdm(population):
            proxy_mem, proxy_latency = self._get_proxy_memory_latency(p)

            if not self.use_remote_benchmark:
                p.metadata['latency'], p.metadata['memory'] = proxy_latency, proxy_mem
            else:
                p.metadata['proxy_latency'], p.metadata['proxy_memory'] = proxy_latency, proxy_mem
                
                # Checks if this architecture was already benchmarked before
                if p.metadata['archid'] not in self.remote_benchmark:
                    cache_misses += 1
                    self.remote_benchmark.send_model(p)
        if self.use_remote_benchmark:
            logger.info(f'{len(population) - cache_misses} benchmark cache hits')

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

        pop_to_eval = [
            p for p in population
            if p.metadata['archid'] not in self.eval_cache
        ]

        if len(pop_to_eval) < len(population):
            logger.info(
                f'{len(population) - len(pop_to_eval)} evaluation cache hits'
            )

        for p in pop_to_eval:
            # create a ray actor per model to be trained
            actor_ref = self._create_training_job(p)
            # create a folder name for the model training logs
            run_path = os.path.join(save_folder, str(p.metadata['archid']))
            os.makedirs(run_path, exist_ok=True)
            # fit and validate the model 
            fit_refs.append(actor_ref.fit_and_validate.remote(run_path=run_path))
        
        # gather all results for all models
        results = ray.get(fit_refs)

        # Cached results
        for p in population:
            if p.metadata['archid'] in self.eval_cache:
                p.metadata['f1'] = self.eval_cache[p.metadata['archid']]

        # Evaluation results
        for r, p in zip(results, pop_to_eval):
            p.metadata['f1'] = r
            self.eval_cache[p.metadata['archid']] = r

    @overrides
    def on_calc_task_accuracy_end(self, current_pop: List[ArchWithMetaData]) -> None:
        if self.use_remote_benchmark:
            evaluated = set()
            nb_tries = 0
            logger.info('Gathering remote benchmark results...')
            pbar = tqdm(total=len(current_pop), desc='Gathering remote benchmark results...')

            while len(evaluated) < len(current_pop) and nb_tries < self.patience:
                for i, p in enumerate(current_pop):
                    # Gets the metrics for all the models in `current_pop``.
                    # we don't need to worry about the cost of checking the same model
                    # more than once since the cost of `get_entity` is infimal
                    # and we may get better estimates for the latency mean when we 
                    # check the same model again (given how the pipeline is constructed)

                    metrics = self.remote_benchmark.get_entity(
                        str(p.metadata['archid'])
                    )

                    # Updates the metadata with the remote benchmark metrics
                    if 'mean' in metrics and metrics['mean']:
                        p.metadata['latency'] = metrics['mean']
                        p.metadata['memory'] = p.metadata['proxy_memory']
                        
                        if i not in evaluated:
                            evaluated.add(i)
                            pbar.update()
                            nb_tries = 0

                    # Resets an entry from the Azure table if the status="complete" prematurely
                    if i not in evaluated and 'status' in metrics and metrics['status'] == 'complete':
                        metrics['status'] = 'incomplete'
                        
                        if 'mean' in metrics:
                            del metrics['mean']
                        
                        if 'total_inference_avg' in metrics:
                            del metrics['total_inference_avg']

                        self.remote_benchmark.update_entity(str(p.metadata['archid']), metrics)

                if len(evaluated) < len(current_pop):
                    pbar.set_description('Sleeping...')
                    logger.info(
                        'Waiting remote benchmark results for '
                        f'{len(current_pop) - len(evaluated)} models...'
                    )
                    time.sleep(self.check_interval)
                    nb_tries += 1

            if nb_tries == self.patience:
                logger.warn('Patience reached. Adding missing models to the next iteration...')

                for i, p in enumerate(current_pop):
                    if i not in evaluated:
                        # Removes possibly incomplete results
                        p.metadata.pop('latency', None)
                        p.metadata.pop('memory', None)

                        # Removes entry from the Azure table
                        self.remote_benchmark.delete_model(p.metadata['archid'])
                        self.models_with_missing_results.append(p)
                
                # Removes the models from the current population
                for p in self.models_with_missing_results:
                    current_pop.remove(p)

            logger.info('Finished gathering remote benchmark results.')

    @overrides
    def on_search_iteration_start(self, current_pop: List[ArchWithMetaData]) -> None:
        if self.use_remote_benchmark and self.models_with_missing_results:
            logger.info(f'Adding missing models to the next iteration...')
            current_pop.extend(self.models_with_missing_results)
            self.models_with_missing_results = []

    def _create_training_job(self, arch:ArchWithMetaData)->List:
        ''' Creates a ray actor that will train a single architecture '''
        # region config
        self.evaluate_for_steps = self.conf_train['evaluate_for_steps']  
        self.val_check_interval = self.conf_train['val_check_interval']  
        self.val_size = self.conf_train['val_size']
        self.img_size = self.conf_train['img_size']
        self.augmentation = self.conf_train['augmentation']
        self.lr = self.conf_train['lr']
        self.lr_exp_decay_gamma = self.conf_train['lr_exp_decay_gamma']
        self.criterion_name = self.conf_train['criterion_name']
        self.batch_size = self.conf_loader['batch_size']
        self.seed = get_conf_common()['seed']
        # region

        # train
        dataset_dir = os.path.join(self.dataroot, 'face_synthetics')
        trainer = ray.remote(
            num_gpus=self.conf_train['gpus_per_job']
        )(SegmentationTrainer)

        ref = trainer.remote(
            arch.arch, dataset_dir=dataset_dir, 
            max_steps=self.evaluate_for_steps,
            val_check_interval=self.val_check_interval,
            val_size=self.val_size, img_size=self.img_size,
            augmentation=self.augmentation,
            batch_size=self.batch_size, lr=self.lr,
            lr_exp_decay_gamma=self.lr_exp_decay_gamma,
            criterion_name=self.criterion_name, seed=self.seed)
        return ref

    @overrides
    def update_pareto_frontier(self, population:List[ArchWithMetaData])->List[ArchWithMetaData]:
        valid_population = [
            p for p in population 
            if all(k in p.metadata and p.metadata[k] for k in ['latency', 'memory', 'f1'])
        ]

        # need all decreasing or increasing quantities
        objs = [
            [1.0 - p.metadata['f1'] for p in valid_population],
            [p.metadata['latency']  for p in valid_population],
            [p.metadata['memory'] for p in valid_population]
        ]
        
        objs = [np.array(obj).reshape(-1, 1) for obj in objs]

        points = np.concatenate(objs, axis=1)
        points_idx = find_pareto_frontier_points(points, is_decreasing=True)
        pareto_points = [valid_population[idx] for idx in points_idx]

        # save all the pareto points
        self._save_yaml(pareto_points, basename='pareto')

        return pareto_points

    def _save_yaml(self, points:List[ArchWithMetaData], basename='pareto')->None:
        exp_dir = utils.full_path(get_expdir())
        save_folder = os.path.join(exp_dir, f'{basename}_iter_{self.iter_num}')
        os.makedirs(save_folder, exist_ok=True)
        for p in points:
            this_name = os.path.join(save_folder, str(p.metadata['archid']) + '.yaml')
            p.arch.to_file(this_name)

    @overrides
    def mutate_parents(self, parents:List[ArchWithMetaData], mutations_per_parent: int = 1)->List[ArchWithMetaData]:
        ''' Using the nearest neighbors as mutations'''
        mutations = {}
        oversample_factor = (
            self.crowd_sorting['oversampling_factor'] if self.crowd_sorting['mutation']
            else 1
        )

        for p in parents:
            candidates = {}
            nb_tries = 0
            patience = 20
            max_latency, max_memory = [self.objectives[k]['max'] for k in ['latency', 'memory']]

            if p.metadata['latency'] > max_latency or p.metadata['memory'] > max_memory:
                logger.info(
                    f'Model {p.metadata["archid"]} has latency {p.metadata["latency"]}'
                    f' or memory {p.metadata["memory"]} that is too high. Skipping mutation.'
                )

                continue

            while len(candidates) < (mutations_per_parent * oversample_factor) and nb_tries < patience:
                for nbr in self.search_space.get_neighbors(p):
                    if nbr.metadata['archid'] not in self.eval_cache:
                        nbr.metadata['generation'] = self.iter_num
                        candidates[nbr.metadata['archid']] = nbr
                nb_tries += 1
            
            if self.crowd_sorting['mutation']:
                candidates_list = list(candidates.items())

                proxy_mem_latency = np.array([
                    list(self._get_proxy_memory_latency(p)) for _, p in candidates_list
                ])

                crowd_dist = compute_crowding_distance(proxy_mem_latency)
                
                # Deletes mutations that are not on the top k
                for idx in np.argsort(-crowd_dist, axis=None)[mutations_per_parent:]:
                    del candidates[candidates_list[idx][0]]

            mutations.update(candidates)

        return list(mutations.values())

    @overrides
    def crossover_parents(self, parents:List[ArchWithMetaData], num_crossovers: int = 1)->List[ArchWithMetaData]:
        # Randomly samples k distinct pairs from `parents`
        children, children_hashes = [], set()

        if len(parents) >= 2:
            pairs = [random.sample(parents, 2) for _ in range(num_crossovers)]

            for p1, p2 in pairs:
                child = self.search_space.crossover(p1, p2)

                if child and child.metadata['archid'] not in children_hashes:
                    children.append(child)
                    children_hashes.add(child.metadata['archid'])

        return children


    @overrides
    def plot_search_state(self, all_pop:List[ArchWithMetaData], pareto:List[ArchWithMetaData], iter_num:int) -> None:
        expdir = Path(get_expdir())
        save_3d_pareto_plot(all_pop, pareto, ['f1', 'latency', 'memory'], iter_num, expdir)
        
        status_df = get_search_status_df(
            all_pop, pareto, iter_num, fields=['archid', 'f1', 'latency', 'memory', 'generation']
        )

        save_2d_pareto_evolution_plot(
            status_df, x='latency', y='f1', save_path=expdir / 'latency_f1_2d_pareto.png',
            x_increasing=False, max_x=self.objectives['latency']['max'], y_increasing=True, max_y=1.0
        )

        save_2d_pareto_evolution_plot(
            status_df, x='memory', y='f1', save_path=expdir / 'memory_f1_2d_pareto.png',
            x_increasing=False, max_x=self.objectives['memory']['max'], y_increasing=True, max_y=1.0
        )


    @overrides
    def save_search_status(self, all_pop:List[ArchWithMetaData], pareto:List[ArchWithMetaData], iter_num:int) -> None:
        fields = [
            'archid', 'f1', 'latency', 'memory', 'proxy_latency', 
            'proxy_memory', 'parent', 'parents', 'macs', 'generation'
        ]

        status_df = get_search_status_df(all_pop, pareto, iter_num, fields)

        # Adds pareto hypervolume
        pareto_points = np.array([
            [p.metadata['latency'], p.metadata['memory'], 1 - p.metadata['f1']]
            for p in pareto
        ])
        
        status_df['pareto_hypervolume'] = compute_pareto_hypervolume(
            pareto_points, 
            np.array([self.objectives['latency']['max'], self.objectives['memory']['max'], 1.0], dtype=np.float32)
        )

        expdir = Path(get_expdir())
        status_df.to_csv(expdir / f'search_status_{iter_num}.csv')
