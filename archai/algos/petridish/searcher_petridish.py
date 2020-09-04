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
from enum import Enum
import copy

# latest verion of ray works on Windows as well
import ray

from overrides import overrides

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
from archai.nas.model_desc import ConvMacroParams, CellDesc, CellType, OpDesc, \
                                  EdgeDesc, TensorShape, TensorShapes, NodeDesc, ModelDesc
from archai.common.trainer import Trainer
from archai.datasets import data
from archai.nas.model import Model
from archai.common.metrics import Metrics
from archai.common import utils
from archai.nas.finalizers import Finalizers
from archai.algos.petridish.petridish_geometry import _convex_hull_from_points
from archai.nas.searcher import SearchResult
from archai.nas.search_combinations import SearchCombinations
from archai.nas.model_desc_builder import ModelDescBuilder



class JobStage(Enum):
    # below values must be assigned in sequence so getting next job stage enum is easy
    SEED = 1
    SEED_TRAINED = 2
    SEARCH = 3
    SEARCH_TRAINED = 4

class ConvexHullPoint:
    _id = 0
    def __init__(self, job_stage:JobStage, parent_id:int,
                 sampling_count:int,
                 model_desc:ModelDesc,
                 metrics:Optional[Metrics]=None,
                 model_stats:Optional[tw.ModelStats]=None) -> None:
        # we only record points after training
        self.job_stage = job_stage
        self.parent_id = parent_id
        self.sampling_count = sampling_count
        self.model_desc = model_desc
        self.metrics = metrics
        self.model_stats = model_stats

        ConvexHullPoint._id += 1
        self.id = ConvexHullPoint._id

    def is_trained_stage(self)->bool:
        return self.job_stage==JobStage.SEARCH_TRAINED or self.job_stage==JobStage.SEED_TRAINED

    def next_stage(self)->JobStage:
        return JobStage(self.job_stage.value+1)

class SearcherPetridish(SearchCombinations):
    def __init__(self):
        super().__init__()

        # initialize ray for distributed training
        ray.init()
        self.num_cpus = ray.nodes()[0]['Resources']['CPU']
        self.num_gpus = ray.nodes()[0]['Resources']['GPU']
        logger.info(f'ray detected {self.num_cpus} cpus and {self.num_gpus} gpus')

    @overrides
    def search(self, conf_search:Config, model_desc_builder:ModelDescBuilder,
                 trainer_class:TArchTrainer, finalizers:Finalizers)->SearchResult:

        # region config vars
        self.conf_search = conf_search
        conf_checkpoint = conf_search['checkpoint']
        resume = conf_search['resume']

        conf_post_train = conf_search['post_train']
        final_desc_foldername = conf_search['final_desc_foldername']

        conf_petridish = conf_search['petridish']
        # petridish distributed search related parameters
        self._convex_hull_eps = conf_petridish['convex_hull_eps']
        self._sampling_max_try = conf_petridish['sampling_max_try']
        self._max_madd = conf_petridish['max_madd']
        self._max_hull_points = conf_petridish['max_hull_points']
        self._checkpoints_foldername = conf_petridish['checkpoints_foldername']
        # endregion

        self._checkpoint = nas_utils.create_checkpoint(conf_checkpoint, resume)

        # parent models list
        self._hull_points: List[ConvexHullPoint] = []

        # checkpoint will restore the hull we had
        is_restored = self._restore_checkpoint()

        # seed the pool with many different seed models of different
        # macro parameters like number of cells, reductions etc if parent pool
        # could not be restored and/or this is the first time this job has been run.
        future_ids = [] if is_restored else  self._create_seed_jobs(conf_search,
                                                                    model_desc_builder)

        while not self._is_search_done():
            logger.info(f'Ray jobs running: {len(future_ids)}')

            if future_ids:
                # get first completed job
                job_id_done, future_ids = ray.wait(future_ids)
                hull_point = ray.get(job_id_done[0])

                logger.info(f'Hull point id {hull_point.id} with stage {hull_point.job_stage.name} completed')

                if hull_point.is_trained_stage():
                    self._update_convex_hull(hull_point)

                    # initiate search on this point
                    sampled_point = self._sample_from_hull()

                    future_id = self.search_model_desc_dist.remote(
                        conf_search, sampled_point, model_desc_builder, trainer_class,
                        finalizers, common.get_state())
                    future_ids.append(future_id)
                    logger.info(f'Added sampled point {sampled_point.id} for search')
                elif hull_point.job_stage==JobStage.SEARCH:
                    # create the job to train the searched model
                    future_id = self.train_model_desc_dist.remote(
                        conf_post_train, hull_point, common.get_state())
                    future_ids.append(future_id)
                    logger.info(f'Added sampled point {hull_point.id} for post-search training')
                else:
                    raise RuntimeError(f'Job stage "{hull_point.job_stage}" is not expected in search loop')

        self._plot_frontier()
        best_point = self._save_frontier(final_desc_foldername)

        # return best point as search result
        search_result = SearchResult(best_point.model_desc, search_metrics=None,
                                     train_metrics=best_point.metrics)
        self.clean_log_result(conf_search, search_result)

        return search_result

    @ray.remote(num_gpus=1)
    def search_model_desc_dist(self, conf_search:Config,
        hull_point:ConvexHullPoint, model_desc_builder:ModelDescBuilder,
        trainer_class:TArchTrainer, finalizers:Finalizers, common_state:CommonState)\
            ->ConvexHullPoint:

        # as this runs in different process, initiaze globals
        common.init_from(common_state)

        assert hull_point.is_trained_stage()

        # cloning is strickly not needed but just in case if we run this
        # function in same process, it would be good to avoid surprise
        model_desc = hull_point.model_desc.clone()
        self._add_node(model_desc, model_desc_builder)

        model_desc, search_metrics = self.search_model_desc(conf_search,
            model_desc, trainer_class, finalizers)

        new_point = ConvexHullPoint(JobStage.SEARCH, hull_point.id,
                                    hull_point.sampling_count,
                                    model_desc, metrics=search_metrics)
        return new_point

    @ray.remote(num_gpus=1)
    def train_model_desc_dist(self, conf_train:Config,
                              hull_point:ConvexHullPoint, common_state:CommonState)\
            ->ConvexHullPoint:
        # as this runs in different process, initiaze globals
        common.init_from(common_state)

        model_metrics = self.train_model_desc(hull_point.model_desc, conf_train)
        model_stats = nas_utils.get_model_stats(model_metrics.model)

        assert not hull_point.is_trained_stage()
        new_point = ConvexHullPoint(hull_point.next_stage(), hull_point.id, hull_point.
                                    sampling_count, hull_point.model_desc, model_metrics.metrics, model_stats)

        return new_point

    def _add_node(self, model_desc:ModelDesc, model_desc_builder:ModelDescBuilder)->None:
        for ci, cell_desc in enumerate(model_desc.cell_descs()):
            reduction = (cell_desc.cell_type==CellType.Reduction)

            nodes = cell_desc.nodes()

            # petridish must seed with one node
            assert len(nodes) > 0
            # input/output channels for all nodes are same
            conv_params = nodes[0].conv_params

            # assign input IDs to nodes, s0 and s1 have IDs 0 and 1
            # however as we will be inserting new node before last one
            # ids are shifted by 2 so previous node IDs are (2+len -2)
            input_ids = list(range(len(nodes)))
            assert len(input_ids) >= 2 # 2 stem inputs + 1 existing node
            op_desc = OpDesc('petridish_reduction_op' if reduction else 'petridish_normal_op',
                                params={
                                    'conv': conv_params,
                                    # specify strides for each input, later we will
                                    # give this to each primitive
                                    '_strides':[2 if reduction and j < 2 else 1 \
                                            for j in input_ids],
                                }, in_len=len(input_ids), trainables=None, children=None)
            edge = EdgeDesc(op_desc, input_ids=input_ids)
            new_node = NodeDesc(edges=[edge], conv_params=conv_params)
            nodes.insert(len(nodes)-1, new_node)

            # output shape of all nodes are same
            node_shapes = cell_desc.node_shapes
            new_node_shape = copy.deepcopy(node_shapes[-1])
            node_shapes.insert(len(node_shapes)-1, new_node_shape)

            # post op needs rebuilding because number of inputs to it has changed so input/output channels may be different
            post_op_shape, post_op_desc = model_desc_builder.build_cell_post_op(cell_desc.stem_shapes,
            node_shapes, cell_desc.conf_cell, ci)
            cell_desc.reset_nodes(nodes, node_shapes,
                                  post_op_desc, post_op_shape)


    def _model_descs_on_front(self, lower_hull:bool=False)\
            ->Tuple[List[ConvexHullPoint], List[ConvexHullPoint], List[float], List[float]]:
        assert(len(self._hull_points) > 0)

        xs = [point.model_stats.MAdd for point in self._hull_points]
        ys = [1.0-point.metrics.best_val_top1() if lower_hull else point.metrics.best_val_top1()
              for point in self._hull_points]

        hull_indices, eps_indices = _convex_hull_from_points(xs, ys, eps=self._convex_hull_eps)
        eps_points = [self._hull_points[i] for i in eps_indices]
        front_points = [self._hull_points[i] for i in hull_indices]

        return front_points, eps_points, xs, ys

    def _plot_frontier(self)->None:
        front_points, eps_points, xs, ys = self._model_descs_on_front(lower_hull=True)

        # save a plot of the convex hull to aid debugging

        hull_xs = [p.model_stats.MAdd for p in eps_points]
        hull_ys = [1.0-p.metrics.best_val_top1() for p in eps_points]
        bound_xs = [p.model_stats.MAdd for p in front_points]
        bound_ys = [(1.0-p.metrics.best_val_top1()) * (1+self._convex_hull_eps) \
                    for p in front_points]

        plt.clf()
        plt.plot(bound_xs, bound_ys, c='red', label='eps-bound')
        plt.scatter(xs, ys, label='pts')
        plt.scatter(hull_xs, hull_ys, c='black', marker='+', label='eps-hull')
        plt.xlabel('Multiply-Additions')
        plt.ylabel('Top1 Error')
        expdir = common.get_expdir()
        assert expdir
        plt.savefig(os.path.join(expdir, 'convex_hull.png'),
            dpi=plt.gcf().dpi, bbox_inches='tight')

    def _save_frontier(self, final_desc_foldername:str)->ConvexHullPoint:
        # make folder to save gallery of models after search
        final_desc_path = utils.full_path(final_desc_foldername, create=True)

        # save the entire gallery of models on the convex hull for evaluation
        front_points, eps_points, xs, ys = self._model_descs_on_front()
        for i, eps_point in enumerate(eps_points):
            savename = os.path.join(self.final_desc_path, f'petridish_{i}.yaml')
            eps_point.model_desc.save(savename)

        # return last model as best performing
        return eps_points[-1]

    def _sample_from_hull(self)->ConvexHullPoint:
        front_points, eps_points, xs, ys = self._model_descs_on_front(lower_hull=True)

        logger.info(f'num models in pool: {len(self._hull_points)}')
        logger.info(f'num models on front: {len(front_points)}')
        logger.info(f'num models on front with eps: {len(eps_points)}')

        # reverse sort by metrics performance
        eps_points.sort(reverse=True, key=lambda p:p.metrics.best_val_top1())

        # default choice
        sampled_point = random.choice(self._hull_points)
        # go through sorted list of models near convex hull
        for _ in range(self._sampling_max_try):
            for point in eps_points:
                p = 1.0 / (point.sampling_count + 1.0)
                should_select = np.random.binomial(1, p)
                if should_select == 1:
                    sampled_point = point

        # if here, sampling was not successful
        logger.warn('sampling was not successful, returning a random parent')

        sampled_point.sampling_count += 1

        return sampled_point

    def _is_search_done(self)->bool:
        '''Terminate search if max MAdd or number of points exceeded'''
        if not self._hull_points:
            return False

        max_madd_parent = max(self._hull_points, key=lambda p:p.model_stats.MAdd)
        return max_madd_parent.model_stats.MAdd > self._max_madd or \
                len(self._hull_points) > self._max_hull_points

    def _create_seed_jobs(self, conf_search:Config, model_desc_builder:ModelDescBuilder)->list:
        conf_model_desc = conf_search['model_desc']
        conf_seed_train = conf_search['seed_train']

        future_ids = [] # ray job IDs
        macro_combinations = list(self.get_combinations(conf_search))
        for reductions, cells, nodes in macro_combinations:
            # if N R N R N R cannot be satisfied, ignore combination
            if cells < reductions * 2 + 1:
                continue

            # create seed model
            model_desc = self.build_model_desc(model_desc_builder,
                                               conf_model_desc,
                                               reductions, cells, nodes)

            # pre-train the seed model
            future_id = self.train_model_desc_dist.remote(JobStage.SEED,
                conf_seed_train, model_desc, common.get_state())

            future_ids.append(future_id)

        return future_ids

    def _update_convex_hull(self, new_point:ConvexHullPoint)->None:
        assert new_point.is_trained_stage() # only add models for which we have metrics and stats
        self._hull_points.append(new_point)

        if self._checkpoint is not None:
            self._checkpoint.new()
            self._checkpoint['convex_hull_points'] = self._hull_points
            self._checkpoint.commit()

        logger.info(f'Added to convex hull points: MAdd {new_point.model_stats.MAdd}, '
                    f'num cells {len(new_point.model_desc.cell_descs())}, '
                    f'num nodes in cell {len(new_point.model_desc.cell_descs()[0].nodes())}')

    def _restore_checkpoint(self)->bool:
        can_restore = self._checkpoint is not None \
                        and 'convex_hull_points' in self._checkpoint
        if can_restore:
            self._hull_points = self._checkpoint['convex_hull_points']
            logger.warn({'Hull restored': True})

        return can_restore

    @overrides
    def build_model_desc(self, model_desc_builder:ModelDescBuilder,
                         conf_model_desc:Config,
                         reductions:int, cells:int, nodes:int)->ModelDesc:
        # reset macro params in copy of config
        conf_model_desc = copy.deepcopy(conf_model_desc)
        conf_model_desc['n_reductions'] = reductions
        conf_model_desc['n_cells'] = cells

        # create model desc for search using model config
        # we will build model without call to model_desc_builder for pre-training
        model_desc = model_desc_builder.build(conf_model_desc, template=None)

        return model_desc