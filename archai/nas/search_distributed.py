# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Iterator, Mapping, Type, Optional, Tuple, List
import math
import copy
import random
import os
from queue import Queue
import secrets
import string

# only works on linux
import ray

import numpy as np
import matplotlib.pyplot as plt

import torch
import tensorwatch as tw
from torch.utils.data.dataloader import DataLoader
import yaml

from archai.common import common
from archai.common.common import logger, CommonState
from archai.common.checkpoint import CheckPoint
from archai.common.config import Config
from archai.nas.arch_trainer import TArchTrainer
from archai.nas import nas_utils
from archai.nas.model_desc import CellType, ModelDesc
from archai.common.trainer import Trainer
from archai.datasets import data
from archai.nas.model import Model
from archai.common.metrics import EpochMetrics, Metrics
from archai.common import utils
from archai.nas.finalizers import Finalizers
from archai.algos.petridish.petridish_geometry import _convex_hull_from_points



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
        self.num_sampled = 0 # REVIEW: ss: what is the use of this variable?

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
def search_desc(model_desc_wrapped, cell_builder, trainer_class, finalizers, conf_train, conf_loader, common_state):
    ''' Remote function which does petridish candidate initialization '''
    common.init_from(common_state)
    logger.pushd('arch_search')

    model_desc = model_desc_wrapped.model_desc
    assert model_desc_wrapped.is_init == True

    # get data
    train_dl, val_dl, _ = data.get_data(conf_loader)
    assert train_dl is not None

    # NOTE: this is warm starting because you are not calling clear_trainables() on the model_desc
    search_iter = -1
    nas_utils.build_cell(model_desc, cell_builder, search_iter)

    model = nas_utils.model_from_desc(model_desc,
                                        droppath=False,
                                        affine=False)

    # search arch
    arch_trainer = trainer_class(conf_train, model, checkpoint=None)
    train_metrics = arch_trainer.fit(train_dl, val_dl)

    metrics_stats = SearchDistributed._create_metrics_stats(
        model, train_metrics, finalizers)
    found_desc = metrics_stats.model_desc

    logger.popd()

    # wrap again
    found_desc_wrapped = ModelDescWrapper(found_desc, model_desc_wrapped.is_init)

    return found_desc_wrapped, metrics_stats


@ray.remote(num_gpus=1)
def train_desc(model_desc_wrapped, conf_train: Config, finalizers: Finalizers, common_state: CommonState) -> Tuple[ModelDesc, MetricsStats]:
    """Train given description"""
    common.init_from(common_state)
    # region conf vars
    conf_trainer = conf_train['trainer']
    conf_loader = conf_train['loader']
    trainer_title = conf_trainer['title']
    epochs = conf_trainer['epochs']
    drop_path_prob = conf_trainer['drop_path_prob']
    # endregion

    logger.pushd(trainer_title)

    train_dl, val_dl, _ = data.get_data(conf_loader)

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
    return model_desc_wrapped, metrics_stats


class SearchDistributed:
    def __init__(self, conf_search: Config, cell_builder: Optional[CellBuilder],
                 trainer_class: TArchTrainer, finalizers: Finalizers) -> None:
        # region config vars
        conf_checkpoint = conf_search['checkpoint']
        self._resume = conf_search['resume']
        self.conf_model_desc = conf_search['model_desc']
        self.conf_loader = conf_search['loader']
        self.conf_train = conf_search['trainer']
        self.final_desc_filename = conf_search['final_desc_filename']
        self.full_desc_filename = conf_search['full_desc_filename']
        self.final_desc_foldername = conf_search['final_desc_foldername']
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
        self.conf_petridish = conf_search['petridish']
        # endregion

        self.cell_builder = cell_builder
        self.trainer_class = trainer_class
        self.finalizers = finalizers
        self._data_cache = {}
        self._parito_filepath = utils.full_path(pareto_summary_filename)
        self._checkpoint = nas_utils.create_checkpoint(conf_checkpoint, self._resume)

        # TODO: why is it still saving without the new() call?
        # initialize the checkpoint dictionary
        # self._checkpoint.new()

        # petridish distributed search related parameters
        self._convex_hull_eps = self.conf_petridish['convex_hull_eps']
        self._max_parent_samples = self.conf_petridish['max_parent_samples']
        self._max_madd = self.conf_petridish['max_madd']
        self._max_parent_models = self.conf_petridish['max_parent_models']
        self._checkpoints_foldername = self.conf_petridish['checkpoints_foldername']

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
        self.num_cpus = ray.nodes()[0]['Resources']['CPU']
        self.num_gpus = ray.nodes()[0]['Resources']['GPU']
        logger.info(f'ray detected {self.num_cpus} cpus and {self.num_gpus} gpus')

        # make folder to save gallery of models after search
        self.final_desc_path = utils.full_path(self.final_desc_foldername, create=True)

        # make folder to save parent pool checkpoints to aid recovery
        self._checkpoints_foldername = utils.full_path(self._checkpoints_foldername, create=True)

        # parent models list
        self._parent_models: List[Tuple[ModelDesc, Optional[MetricsStats]]] = []


    def _get_seed_model_desc(self) -> Tuple[int, int, int]:
        return self.base_reductions, self.base_cells, self.base_nodes


    def _get_models_near_convex_hull(self):
        assert(len(self._parent_models) > 0)

        xs = []
        ys = []
        for _, metrics_stats in self._parent_models:
            xs.append(metrics_stats.model_stats.MAdd) # REVIEW: ss: why MAdd?
            ys.append(metrics_stats.best_metrics().top1.avg)

        _, eps_indices = _convex_hull_from_points(xs, ys, eps=self._convex_hull_eps)
        eps_models = [self._parent_models[i][0] for i in eps_indices]
        return eps_models


    def _sample_model_from_parent_pool(self):
        assert(len(self._parent_models) > 0)

        xs = []
        ys = []
        for _, metrics_stats in self._parent_models:
            xs.append(metrics_stats.model_stats.MAdd)
            # ys have to be error as we are computing lower convex hull
            ys.append(1.0 - metrics_stats.best_metrics().top1.avg)

        hull_indices, eps_indices = _convex_hull_from_points(xs, ys, eps=self._convex_hull_eps)

        logger.info(f'num models in parent pool: {len(self._parent_models)}')
        logger.info(f'num models near pareto: {len(eps_indices)}')

        # reverse sort by performance
        x_y_num_sampled = [(xs[i], ys[i], self._parent_models[i][1].num_sampled) for i in eps_indices]
        x_y_num_sampled.sort(reverse=True, key=lambda x:x[1])

        # save a plot of the convex hull to aid debugging

        hull_xs = [xs[i] for i in eps_indices]
        hull_ys = [ys[i] for i in eps_indices]
        bound_xs = [xs[i] for i in hull_indices]
        bound_ys = [ys[i] * (1+self._convex_hull_eps) for i in hull_indices]
        plt.clf()
        plt.plot(bound_xs, bound_ys, c='red', label='eps-bound')
        plt.scatter(xs, ys, label='pts')
        plt.scatter(hull_xs, hull_ys, c='black', marker='+', label='eps-hull')
        plt.xlabel('Multiply-Additions')
        plt.ylabel('Top1 Error')
        expdir = common.get_expdir()
        plt.savefig(os.path.join(expdir, 'convex_hull.png'),
            dpi=plt.gcf().dpi, bbox_inches='tight')

        # go through sorted list of models near convex hull
        counter = 0
        while(counter < self._max_parent_samples):
            counter += 1
            for i, (_, _, num_sampled) in enumerate(x_y_num_sampled):
                p = 1.0 / (num_sampled + 1.0)
                should_select = np.random.binomial(1, p)
                if should_select == 1:
                    return self._parent_models[i]

        # if here, sampling was not successful
        logger.warn('sampling was not successful, returning a random parent')
        ind = random.randint(0, len(self._parent_models)-1)
        return self._parent_models[ind]


    def _should_terminate_search(self)->bool:
        ''' Looks at the parent pool and decides whether to terminate search '''
        if not self._parent_models:
            return False

        max_madd_parent = max(self._parent_models, key=lambda x: x[1].model_stats.MAdd)
        if max_madd_parent[1].model_stats.MAdd > self._max_madd or len(self._parent_models) > self._max_parent_models:
            return True
        else:
            return False


    def _macro_combinations(self)->Iterator[Tuple[int, int, int]]:
        if not self.pareto_enabled:
            yield self.base_reductions, self.base_cells, self.base_nodes
        else:
            # TODO: what happens when reductions is 3 but cells is 2? have to step
            # through code and check
            for reductions in range(self.base_reductions, self.max_reductions+1):
                for cells in range(self.base_cells, self.max_cells+1):
                    for nodes in range(self.base_nodes, self.max_nodes+1):
                        yield reductions, cells, nodes


    def search_loop(self)->None:
        # TODO: I noticed that metrics_stats already includes model_desc. For saving and restoring
        # do we need to save model_desc separately or just saving the metrics_stats is enough?
        # Do we need to have model_desc separately in the parent pool? is metrics_stats enough?

        # the pool of ray job ids
        future_ids = []

        # attempt to restore parent pool
        parent_pool_restored = self._restore_parent_pool()

        # seed the pool with many different seed models of different
        # macro parameters like number of cells, reductions etc if parent pool
        # could not be restored and/or this is the first time this job has been run.
        if not parent_pool_restored:
            macro_combinations = list(self._macro_combinations())
            for reductions, cells, nodes in macro_combinations:
                # if N R N R N R cannot be satisfied, ignore combination
                if cells < reductions * 2 + 1:
                    continue

                model_desc = self._build_macro(reductions, cells, nodes)

                if self.cell_builder:
                    self.cell_builder.seed(model_desc)
                model_desc_wrapped = ModelDescWrapper(model_desc, False)
                this_child_id = train_desc.remote(model_desc_wrapped, self.conf_presearch_train, self.finalizers, common.get_state())

                future_ids.append(this_child_id)

        while not self._should_terminate_search():
            logger.info(f'num jobs currently in pool (waiting or being processed) {len(future_ids)}')

            if future_ids:
                job_id_done, future_ids = ray.wait(future_ids)
                model_desc_wrapped, metrics_stats = ray.get(job_id_done[0])

                # REVIEW: ss: I'm bit confused about role of is_init but lets talk about that on call
                if model_desc_wrapped.is_init:
                    # a model just got initialized
                    # push a job to train it
                    logger.info('model just got initialized.')
                    model_desc_wrapped.is_init = False
                    this_child_id = train_desc.remote(model_desc_wrapped, self.conf_postsearch_train, self.finalizers, common.get_state())
                    future_ids.append(this_child_id)
                else:
                    logger.info('child/seed model just finished training.')
                    # a child job finished.
                    model_desc_wrapped.is_init = True

                    # increase the counter tracking number of times it has been sampled
                    # REVIEW: ss: why not put num_sampled in ModelDescWrapper
                    metrics_stats.num_sampled += 1

                    # add it to the parent models pool
                    self._parent_models.append((model_desc_wrapped.model_desc, metrics_stats))

                    # save to disk for restoring
                    filename = os.path.join(self._checkpoints_foldername, f'parent_desc_{len(self._parent_models)}.yaml')
                    model_desc_wrapped.model_desc.save(filename, save_trainables=True)
                    self._record_checkpoint(filename, metrics_stats)
                    logger.info(f'appended to parent a model with MAdd {metrics_stats.model_stats.MAdd}, num cells {len(model_desc_wrapped.model_desc.cell_descs())}, num nodes in cell {len(model_desc_wrapped.model_desc.cell_descs()[0].nodes())}')

                    # sample a model from parent pool
                    model_desc, _ = self._sample_model_from_parent_pool()
                    model_desc_wrapped = ModelDescWrapper(model_desc, True)
                    this_search_id = search_desc.remote(model_desc_wrapped, self.cell_builder, self.trainer_class, self.finalizers, self.conf_train, self.conf_loader, common.get_state())
                    future_ids.append(this_search_id)
                    logger.info('just added a new model to processing pool')

            # if we are not utilizing all gpus in the system lets sample more
            # models from the parent pool
            num_current_jobs = len(future_ids)
            num_unused_gpus = int(self.num_gpus - num_current_jobs)
            if num_unused_gpus > 0 and len(self._parent_models) > 0:
                for _ in range(num_unused_gpus):
                    # sample a model from parent pool
                    model_desc, _ = self._sample_model_from_parent_pool()
                    model_desc_wrapped = ModelDescWrapper(model_desc, True)
                    this_search_id = search_desc.remote(model_desc_wrapped, search_iter, self.cell_builder, self.trainer_class, self.finalizers, train_dl, val_dl, self.conf_train, common.get_state())
                    future_ids.append(this_search_id)
                    logger.info('just added a new model to processing pool')


        # save the entire gallery of models on the convex hull for evaluation
        eps_models = self._get_models_near_convex_hull()
        for i, eps_model in enumerate(eps_models):
            savename = os.path.join(self.final_desc_path, f'petridish_{i}.yaml')
            eps_model.save(savename)


    def _record_checkpoint(self, model_desc_filename:str, metrics_stats:MetricsStats):
        if self._checkpoint is not None:
            parent_pool_entry = {'model_desc_filename': model_desc_filename,
                                 'metrics_stats': yaml.dump(metrics_stats)}
            this_key = f'parent_pool_entry_{len(self._parent_models)}'
            self._checkpoint[this_key] = parent_pool_entry
            self._checkpoint.commit()


    def _restore_parent_pool(self)->bool:
        # check that the folder exists
        if not os.path.isdir(self._checkpoints_foldername):
            logger.warn(f'checkpoints folder {self._checkpoints_foldername} does not exist. unable to restore parent pool.')
            return False

        # check that the checkpoint is valid
        if nas_utils.checkpoint_empty(self._checkpoint):
            logger.warn(f'checkpoint is empty. unable to restore parent pool.')
            return False

        # get list of all files in the folder
        # onlyfiles = [f for f in os.listdir(self._checkpoints_foldername) if os.path.isfile(os.path.join(self._checkpoints_foldername, f))]

        for key in self._checkpoint.keys():
            state = self._checkpoint.get(key, None)
            if state is not None:
                model_desc_filename = state['model_desc_filename']
                model_desc = ModelDesc.load(model_desc_filename, load_trainables=True)
                metrics_stats = yaml.load(state['metrics_stats'], Loader=yaml.Loader)
                self._parent_models.append((model_desc, metrics_stats))

        return True




    # def _seed_model(self, model_desc, reductions, cells, nodes) -> ModelDesc:
    #     if self.cell_builder:
    #         self.cell_builder.seed(model_desc)
    #     metrics_stats = self._train_desc(model_desc, self.conf_presearch_train)
    #     self._save_trained(reductions, cells, nodes, -1, metrics_stats)
    #     return metrics_stats.model_desc


    def _build_macro(self, reductions: int, cells: int, nodes: int) -> ModelDesc:
        conf_model_desc = copy.deepcopy(self.conf_model_desc)
        conf_model_desc['n_reductions'] = reductions
        conf_model_desc['n_cells'] = cells
        # create model desc for search using model config
        # we will build model without call to cell_builder for pre-training
        model_desc = nas_utils.create_macro_desc(conf_model_desc,
                                                 template_model_desc=None)
        return model_desc


    # def _train_desc(self, model_desc:ModelDesc, conf_train:Config)->MetricsStats:
    #     """Train given description"""
    #     # region conf vars
    #     conf_trainer = conf_train['trainer']
    #     conf_loader = conf_train['loader']
    #     trainer_title = conf_trainer['title']
    #     epochs = conf_trainer['epochs']
    #     drop_path_prob = conf_trainer['drop_path_prob']
    #     # endregion

    #     logger.pushd(trainer_title)

    #     if epochs == 0:
    #         # nothing to pretrain, save time
    #         metrics_stats = MetricsStats(model_desc, None, None)
    #     else:
    #         model = nas_utils.model_from_desc(model_desc,
    #                                           droppath=drop_path_prob>0.0,
    #                                           affine=True)

    #         # get data
    #         train_dl, val_dl = self.get_data(conf_loader)
    #         assert train_dl is not None

    #         trainer = Trainer(conf_trainer, model, checkpoint=None)
    #         train_metrics = trainer.fit(train_dl, val_dl)

    #         metrics_stats = SearchDistributed._create_metrics_stats(model, train_metrics, self.finalizers)

    #     logger.popd()
    #     return metrics_stats


    # def _save_trained(self, reductions: int, cells: int, nodes: int,
    #                   search_iter: int,
    #                   metrics_stats: MetricsStats) -> None:
    #     """Save the model and metric info into a log file"""

    #     # construct path where we will save
    #     subdir = utils.full_path(
    #         self.metrics_dir.format(**vars()), create=True)

    #     # save metric_infi
    #     metrics_stats_filepath = os.path.join(subdir, 'metrics_stats.yaml')
    #     if metrics_stats_filepath:
    #         with open(metrics_stats_filepath, 'w') as f:
    #             yaml.dump(metrics_stats, f)

    #     # save just metrics separately
    #     metrics_filepath = os.path.join(subdir, 'metrics.yaml')
    #     if metrics_filepath:
    #         with open(metrics_filepath, 'w') as f:
    #             yaml.dump(metrics_stats.train_metrics, f)

    #     logger.info({'metrics_stats_filepath': metrics_stats_filepath,
    #                  'metrics_filepath': metrics_filepath})

    #     # append key info in root pareto data
    #     if self._parito_filepath:
    #         train_top1 = val_top1 = train_epoch = val_epoch = math.nan
    #         # extract metrics
    #         if metrics_stats.train_metrics:
    #             best_metrics = metrics_stats.train_metrics.run_metrics.best_epoch()
    #             train_top1 = best_metrics[0].top1.avg
    #             train_epoch = best_metrics[0].index
    #             if best_metrics[1]:
    #                 val_top1 = best_metrics[1].top1.avg if len(
    #                     best_metrics) > 1 else math.nan
    #                 val_epoch = best_metrics[1].index if len(
    #                     best_metrics) > 1 else math.nan

    #         # extract model stats
    #         if metrics_stats.model_stats:
    #             flops = metrics_stats.model_stats.Flops
    #             parameters = metrics_stats.model_stats.parameters
    #             inference_memory = metrics_stats.model_stats.inference_memory
    #             inference_duration = metrics_stats.model_stats.duration
    #         else:
    #             flops = parameters = inference_memory = inference_duration = math.nan

    #         utils.append_csv_file(self._parito_filepath, [
    #             ('reductions', reductions),
    #             ('cells', cells),
    #             ('nodes', nodes),
    #             ('search_iter', search_iter),
    #             ('train_top1', train_top1),
    #             ('train_epoch', train_epoch),
    #             ('val_top1', val_top1),
    #             ('val_epoch', val_epoch),
    #             ('flops', flops),
    #             ('params', parameters),
    #             ('inference_memory', inference_memory),
    #             ('inference_duration', inference_duration)
    #         ])


    # def get_data(self, conf_loader: Config) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    #     # first get from cache
    #     train_ds, val_ds = self._data_cache.get(id(conf_loader), (None, None))
    #     # if not found in cache then create
    #     if train_ds is None:
    #         train_ds, val_ds, _ = data.get_data(conf_loader)
    #         self._data_cache[id(conf_loader)] = (train_ds, val_ds)
    #     return train_ds, val_ds


    @staticmethod
    def _create_metrics_stats(model: Model, train_metrics: Metrics, finalizers: Finalizers) -> MetricsStats:
        # TODO: Dey verify that this is not an issue during petridish distributed training
        finalized = finalizers.finalize_model(model, restore_device=False)
        # model stats is doing some hooks so do it last
        model_stats = tw.ModelStats(model, [1, 3, 32, 32],  # TODO: remove this hard coding
                                    clone_model=True)
        return MetricsStats(finalized, train_metrics, model_stats)
