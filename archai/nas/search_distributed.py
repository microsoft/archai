# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Iterator, Mapping, Type, Optional, Tuple, List
import math
import copy
import random
import os
from queue import Queue

# only works on linux
import ray

import torch
import tensorwatch as tw
from torch.utils.data.dataloader import DataLoader
import yaml

from archai.common.common import logger
from archai.common.checkpoint import CheckPoint
from archai.common.config import Config
from archai.nas.cell_builder import CellBuilder
from archai.nas.arch_trainer import TArchTrainer
from archai.nas import nas_utils
from archai.nas.model_desc import CellType, ModelDesc
from archai.common.trainer import Trainer
from archai.datasets import data
from archai.nas.model import Model
from archai.common.metrics import EpochMetrics, Metrics
from archai.common import utils
from archai.nas.finalizers import Finalizers



class ModelDescWrapper:
    """ Holds a ModelDesc and a flag indicating whether it is an initialization or a child one """
    def __init__(self, model_desc: ModelDesc, is_init: bool):
        self.model_desc = model_desc
        self.is_init = is_init


class MetricsStats:
    """Holds model statistics and training metrics for given description"""

    def __init__(self, model_desc: ModelDesc,
                 train_metrics: Optional[Metrics],
                 model_stats: Optional[tw.ModelStats]) -> None:
        self.model_desc = model_desc
        self.train_metrics = train_metrics
        self.model_stats = model_stats

    def __str__(self) -> str:
        best = self.best_metrics()
        if best is not None:
            return f'top1={best.top1.avg}'
        return 'None'

    def state_dict(self) -> Mapping:
        return {
            'model_desc': self.model_desc.state_dict(),
            'train_metrics': self.train_metrics.state_dict(),
            'model_stats': utils.state_dict(self.model_stats)
        }

    def load_state_dict(self, state_dict: Mapping) -> None:
        self.model_desc.load_state_dict(state_dict['model_desc'])
        self.train_metrics.load_state_dict(state_dict['train_metrics'])
        utils.load_state_dict(self.model_stats, state_dict['model_stats'])

    def best_metrics(self) -> Optional[EpochMetrics]:
        if self.train_metrics is None:
            return None
        best_train, best_val = self.train_metrics.run_metrics.best_epoch()
        return best_train if best_val is None else best_val

    def is_better(self, other: Optional['MetricsStats']) -> bool:
        # if both same or other None, this one is better
        if other is None:
            return True
        best_other = other.best_metrics()
        if best_other is None:
            return True
        best_this = self.best_metrics()
        if best_this is None:
            return False
        return best_this.top1.avg >= best_other.top1.avg


class SearchResult:
    def __init__(self, metrics_stats: MetricsStats,
                 macro_params: Tuple[int, int, int]) -> None:
        self.metrics_stats = metrics_stats
        # macro_params: reductions, cells, nodes
        self.macro_params = macro_params

    def state_dict(self) -> Mapping:
        return {'metrics_stats': self.metrics_stats.state_dict(),
                'macro_params': self.macro_params}

    def load_state_dict(self, state_dict) -> None:
        self.metrics_stats.load_state_dict(state_dict['metrics_stats'])
        self.macro_params = state_dict['macro_params']

    def model_desc(self) -> ModelDesc:
        return self.metrics_stats.model_desc



@ray.remote(num_gpus=1)
def search_desc(model_desc_wrapped, search_iter, cell_builder, trainer_class, finalizers, train_dl, val_dl, conf_train):
    ''' Remote function which does petridish candidate initialization '''
    # TODO: all parallel processes will create this same folder and cause problems in logging
    logger.pushd('arch_search')

    model_desc = model_desc_wrapped.model_desc

    assert model_desc_wrapped.is_init == True

    nas_utils.build_cell(model_desc, cell_builder, search_iter)

    model = nas_utils.model_from_desc(model_desc,
                                        droppath=False,
                                        affine=False)

    # get data
    assert train_dl is not None

    # search arch
    arch_trainer = trainer_class(conf_train, model, checkpoint=None)
    train_metrics = arch_trainer.fit(train_dl, val_dl)

    metrics_stats = SearchDistributed._create_metrics_stats(
        model, train_metrics, finalizers)
    found_desc = metrics_stats.model_desc

    logger.popd()

    return found_desc, metrics_stats


@ray.remote(num_gpus=1)
def train_desc(model_desc_wrapped, conf_train: Config, finalizers: Finalizers, train_dl: DataLoader, val_dl: DataLoader) -> Tuple[ModelDesc, MetricsStats]:
    """Train given description"""
    # region conf vars
    conf_trainer = conf_train['trainer']
    conf_loader = conf_train['loader']
    trainer_title = conf_trainer['title']
    epochs = conf_trainer['epochs']
    drop_path_prob = conf_trainer['drop_path_prob']
    # endregion

    # TODO: 
    logger.pushd(trainer_title)

    model_desc = model_desc_wrapped.model_desc
    assert model_desc_wrapped.is_init == False

    if epochs == 0:
        # nothing to pretrain, save time
        metrics_stats = MetricsStats(model_desc, None, None)
    else:
        model = nas_utils.model_from_desc(model_desc,
                                            droppath=drop_path_prob > 0.0,
                                            affine=True)

        # get data
        assert train_dl is not None

        trainer = Trainer(conf_trainer, model, checkpoint=None)
        train_metrics = trainer.fit(train_dl, val_dl)

        metrics_stats = SearchDistributed._create_metrics_stats(
            model, train_metrics, finalizers)

    logger.popd()
    return model_desc, metrics_stats


class SearchDistributed:
    def __init__(self, conf_search: Config, cell_builder: Optional[CellBuilder],
                 trainer_class: TArchTrainer, finalizers: Finalizers) -> None:
        # region config vars
        conf_checkpoint = conf_search['checkpoint']
        resume = conf_search['resume']
        self.conf_model_desc = conf_search['model_desc']
        self.conf_loader = conf_search['loader']
        self.conf_train = conf_search['trainer']
        self.final_desc_filename = conf_search['final_desc_filename']
        self.full_desc_filename = conf_search['full_desc_filename']
        self.metrics_dir = conf_search['metrics_dir']
        self.conf_presearch_train = conf_search['seed_train']
        self.conf_postsearch_train = conf_search['post_train']
        conf_pareto = conf_search['pareto']
        self.base_cells = self.conf_model_desc['n_cells']
        self.max_cells = conf_pareto['max_cells']
        self.base_reductions = self.conf_model_desc['n_reductions']
        self.max_reductions = conf_pareto['max_reductions']
        self.base_nodes = self.conf_model_desc['n_nodes']
        self.max_nodes = conf_pareto['max_nodes']
        self.search_iters = conf_search['search_iters']
        self.pareto_enabled = conf_pareto['enabled']
        pareto_summary_filename = conf_pareto['summary_filename']
        # endregion

        self.cell_builder = cell_builder
        self.trainer_class = trainer_class
        self.finalizers = finalizers
        self._data_cache = {}
        self._parito_filepath = utils.full_path(pareto_summary_filename)
        self._checkpoint = nas_utils.create_checkpoint(conf_checkpoint, resume)

        logger.info({'pareto_enabled': self.pareto_enabled,
                     'base_reductions': self.base_reductions,
                     'base_cells': self.base_cells,
                     'base_nodes': self.base_nodes,
                     'max_reductions': self.max_reductions,
                     'max_cells': self.max_cells,
                     'max_nodes': self.max_nodes
                     })

        # initialize ray for distributed training
        ray.init()

        # parent models list
        self._parent_models: List[Tuple[ModelDesc, Optional[MetricsStats]]] = []

       
    def _get_seed_model_desc(self) -> Tuple[int, int, int]:
        return self.base_reductions, self.base_cells, self.base_nodes


    def _sample_model_from_parent_pool(self):
        points = []
        for model_desc, metrics_stats in self._parent_models:
            points[]


        # TODO: Right now just random sampling
        # implement proper convex hull based sampling later
        index = random.randint(0, len(self._parent_models) - 1)
        return self._parent_models[index]


    def search_loop(self)->None:
        # get seed model
        reductions, cells, nodes = self._get_seed_model_desc()
        model_desc = self._build_macro(reductions, cells, nodes)
        # prep seed model and add to the parent set
        model_desc = self._seed_model(model_desc, reductions, cells, nodes)
        model_desc_wrapped = ModelDescWrapper(model_desc, True)
        self._parent_models.append((model_desc, None))

        # sample a model from parent pool for initialization phase
        sampled_model_desc_wrapped, _ = self._sample_model_from_parent_pool()

        # seed train that model
        search_iter = -1
        train_dl, val_dl = self.get_data(self.conf_loader)
        future_ids = [search_desc.remote(sampled_model_desc_wrapped, search_iter, self.cell_builder, self.trainer_class, self.finalizers, train_dl, val_dl, self.conf_train)]
    
        while len(future_ids):
            print(f'Num jobs currently in pool (waiting or being processed) {len(future_ids)}')

            job_id_done, future_ids = ray.wait(future_ids)
            model_desc_wrapped, metrics_stats = ray.get(job_id_done[0])

            if model_desc_wrapped.is_init:
                # a model just got initialized
                # push a job to train it
                model_desc_wrapped.is_init = False
                this_child_id = train_desc.remote(model_desc_wrapped, self.conf_train, self.finalizers, train_dl, val_dl)
                future_ids.append(this_child_id) 
            else:
                # a child job finished. 
                # add it to the parent models pool
                model_desc_wrapped.is_init = True
                self._parent_models.append((model_desc_wrapped.model_desc, metrics_stats))
                # sample a model from parent pool
                model_desc, _ = self._sample_model_from_parent_pool()
                model_desc_wrapped = ModelDescWrapper(model_desc, True)
                this_search_id = search_desc.remote(model_desc_wrapped, search_iter, self.cell_builder, self.trainer_class, self.finalizers, train_dl, val_dl, self.conf_train)
                future_ids.append(this_search_id)
            

    def _restore_checkpoint(self, macro_combinations)\
            -> Tuple[int, Optional[SearchResult]]:
        checkpoint_avail = self._checkpoint is not None
        resumed, state = False, None
        start_macro, best_result = 0, None
        if checkpoint_avail:
            state = self._checkpoint.get('search', None)
            if state is not None:
                start_macro = state['start_macro']
                assert start_macro >= 0 and start_macro < len(
                    macro_combinations)
                best_result = yaml.load(
                    state['best_result'], Loader=yaml.Loader)

                start_macro += 1  # resume after the last checkpoint
                resumed = True

        if not resumed:
            # erase previous file left over from run
            utils.zero_file(self._parito_filepath)

        logger.warn({'resumed': resumed, 'checkpoint_avail': checkpoint_avail,
                     'checkpoint_val': state is not None,
                     'start_macro': start_macro,
                     'total_macro_combinations': len(macro_combinations)})
        return start_macro, best_result


    def _record_checkpoint(self, macro_comb_i: int, best_result: SearchResult) -> None:
        if self._checkpoint is not None:
            state = {'start_macro': macro_comb_i,
                     'best_result': yaml.dump(best_result)}
            self._checkpoint.new()
            self._checkpoint['search'] = state
            self._checkpoint.commit()


    def _seed_model(self, model_desc, reductions, cells, nodes) -> ModelDesc:
        if self.cell_builder:
            self.cell_builder.seed(model_desc)
        metrics_stats = self._train_desc(model_desc, self.conf_presearch_train)
        self._save_trained(reductions, cells, nodes, -1, metrics_stats)
        return metrics_stats.model_desc


    def _build_macro(self, reductions: int, cells: int, nodes: int) -> ModelDesc:
        conf_model_desc = copy.deepcopy(self.conf_model_desc)
        conf_model_desc['n_reductions'] = reductions
        conf_model_desc['n_cells'] = cells
        # create model desc for search using model config
        # we will build model without call to cell_builder for pre-training
        model_desc = nas_utils.create_macro_desc(self.conf_model_desc,
                                                 template_model_desc=None)
        return model_desc


    def _train_desc(self, model_desc:ModelDesc, conf_train:Config)->MetricsStats:
        """Train given description"""
        # region conf vars
        conf_trainer = conf_train['trainer']
        conf_loader = conf_train['loader']
        trainer_title = conf_trainer['title']
        epochs = conf_trainer['epochs']
        drop_path_prob = conf_trainer['drop_path_prob']
        # endregion

        logger.pushd(trainer_title)

        if epochs == 0:
            # nothing to pretrain, save time
            metrics_stats = MetricsStats(model_desc, None, None)
        else:
            model = nas_utils.model_from_desc(model_desc,
                                              droppath=drop_path_prob>0.0,
                                              affine=True)

            # get data
            train_dl, val_dl = self.get_data(conf_loader)
            assert train_dl is not None

            trainer = Trainer(conf_trainer, model, checkpoint=None)
            train_metrics = trainer.fit(train_dl, val_dl)

            metrics_stats = SearchDistributed._create_metrics_stats(model, train_metrics, self.finalizers)

        logger.popd()
        return metrics_stats


    def _save_trained(self, reductions: int, cells: int, nodes: int,
                      search_iter: int,
                      metrics_stats: MetricsStats) -> None:
        """Save the model and metric info into a log file"""

        # construct path where we will save
        subdir = utils.full_path(
            self.metrics_dir.format(**vars()), create=True)

        # save metric_infi
        metrics_stats_filepath = os.path.join(subdir, 'metrics_stats.yaml')
        if metrics_stats_filepath:
            with open(metrics_stats_filepath, 'w') as f:
                yaml.dump(metrics_stats, f)

        # save just metrics separately
        metrics_filepath = os.path.join(subdir, 'metrics.yaml')
        if metrics_filepath:
            with open(metrics_filepath, 'w') as f:
                yaml.dump(metrics_stats.train_metrics, f)

        logger.info({'metrics_stats_filepath': metrics_stats_filepath,
                     'metrics_filepath': metrics_filepath})

        # append key info in root pareto data
        if self._parito_filepath:
            train_top1 = val_top1 = train_epoch = val_epoch = math.nan
            # extract metrics
            if metrics_stats.train_metrics:
                best_metrics = metrics_stats.train_metrics.run_metrics.best_epoch()
                train_top1 = best_metrics[0].top1.avg
                train_epoch = best_metrics[0].index
                if best_metrics[1]:
                    val_top1 = best_metrics[1].top1.avg if len(
                        best_metrics) > 1 else math.nan
                    val_epoch = best_metrics[1].index if len(
                        best_metrics) > 1 else math.nan

            # extract model stats
            if metrics_stats.model_stats:
                flops = metrics_stats.model_stats.Flops
                parameters = metrics_stats.model_stats.parameters
                inference_memory = metrics_stats.model_stats.inference_memory
                inference_duration = metrics_stats.model_stats.duration
            else:
                flops = parameters = inference_memory = inference_duration = math.nan

            utils.append_csv_file(self._parito_filepath, [
                ('reductions', reductions),
                ('cells', cells),
                ('nodes', nodes),
                ('search_iter', search_iter),
                ('train_top1', train_top1),
                ('train_epoch', train_epoch),
                ('val_top1', val_top1),
                ('val_epoch', val_epoch),
                ('flops', flops),
                ('params', parameters),
                ('inference_memory', inference_memory),
                ('inference_duration', inference_duration)
            ])


    def get_data(self, conf_loader: Config) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        # first get from cache
        train_ds, val_ds = self._data_cache.get(id(conf_loader), (None, None))
        # if not found in cache then create
        if train_ds is None:
            train_ds, val_ds, _ = data.get_data(conf_loader)
            self._data_cache[id(conf_loader)] = (train_ds, val_ds)
        return train_ds, val_ds


    @staticmethod
    def _create_metrics_stats(model: Model, train_metrics: Metrics, finalizers: Finalizers) -> MetricsStats:
        finalized = finalizers.finalize_model(model, restore_device=False)
        # model stats is doing some hooks so do it last
        model_stats = tw.ModelStats(model, [1, 3, 32, 32],  # TODO: remove this hard coding
                                    clone_model=True)
        return MetricsStats(finalized, train_metrics, model_stats)
