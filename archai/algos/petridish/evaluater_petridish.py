# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, List, Tuple
import importlib
import sys
import string
import os
import copy
import math

# only works on linux
import ray

from overrides import overrides

import tensorwatch as tw

import torch
from torch import nn
import tensorwatch as tw
import yaml
import matplotlib.pyplot as plt

from archai.common.trainer import Trainer
from archai.common.config import Config
from archai.common.common import get_expdir, logger
from archai.datasets import data
from archai.nas.model_desc import CellType, ModelDesc
from archai.nas.model import Model
from archai.nas.model_desc_builder import ModelDescBuilder
from archai.nas import nas_utils
from archai.common import common
from archai.common import ml_utils, utils
from archai.common.metrics import Metrics
from archai.nas.evaluater import Evaluater, EvalResult

class EvaluaterPetridish(Evaluater):

    @overrides
    def evaluate(self, conf_eval:Config, model_desc_builder:ModelDescBuilder)->EvalResult:
        """Takes a folder of model descriptions output by search process and
        trains them in a distributed manner using ray with 1 gpu"""

        logger.pushd('evaluate')

        final_desc_foldername:str = conf_eval['final_desc_foldername']

        # get list of model descs in the gallery folder
        final_desc_folderpath = utils.full_path(final_desc_foldername)
        files = [os.path.join(final_desc_folderpath, f) for f in os.listdir(final_desc_folderpath) if os.path.isfile(os.path.join(final_desc_folderpath, f))]
        logger.info({'models to train':len(files)})

        self._ensure_dataset_download(conf_eval)

        future_ids = []
        for model_desc_filename in files:
            future_id = EvaluaterPetridish._train_dist.remote(self, conf_eval, model_desc_builder,
                                              model_desc_filename, common.get_state())
            future_ids.append(future_id)

        # wait for all eval jobs to be finished
        ready_refs, remaining_refs = ray.wait(future_ids, num_returns=len(future_ids))

        # plot pareto curve of gallery of models
        metric_stats_all = [ray.get(ready_ref) for ready_ref in ready_refs]
        self._plot_model_gallery(metric_stats_all)

        best_metric_stats = max(metric_stats_all, key=lambda ms:ms[0].best_val_top1())
        logger.info({'best_top1_val':best_metric_stats[0].best_val_top1(),
                     'best_MAdd': best_metric_stats[1].MAdd})

        logger.popd()

        return EvalResult(best_metric_stats[0])

    @staticmethod
    @ray.remote(num_gpus=1)
    def _train_dist(evaluater:Evaluater, conf_eval:Config, model_desc_builder:ModelDescBuilder,
                    model_desc_filename:str, common_state)->Tuple[Metrics, tw.ModelStats]:
        """Train given a model"""

        common.init_from(common_state)

        # region config vars
        conf_model_desc = conf_eval['model_desc']
        max_cells = conf_model_desc['n_cells']

        conf_checkpoint = conf_eval['checkpoint']
        resume = conf_eval['resume']

        conf_petridish = conf_eval['petridish']
        cell_count_scale = conf_petridish['cell_count_scale']
        #endregion

        #register ops as we are in different process now
        model_desc_builder.pre_build(conf_model_desc)

        model_filename = utils.append_to_filename(model_desc_filename, '_model', '.pt')
        full_desc_filename = utils.append_to_filename(model_desc_filename, '_full', '.yaml')
        metrics_filename = utils.append_to_filename(model_desc_filename, '_metrics', '.yaml')
        model_stats_filename = utils.append_to_filename(model_desc_filename, '_model_stats', '.yaml')

        # create checkpoint for this specific model desc by changing the config
        checkpoint = None
        if conf_checkpoint is not None:
            conf_checkpoint['filename'] = model_filename.split('.')[0] + '_checkpoint.pth'
            checkpoint = nas_utils.create_checkpoint(conf_checkpoint, resume)

            if checkpoint is not None and resume:
                if 'metrics_stats' in checkpoint:
                    # return the output we had recorded in the checkpoint
                    train_metrics, model_stats = checkpoint['metrics_stats']
                    return train_metrics, model_stats

        # template model is what we used during the search
        template_model_desc = ModelDesc.load(model_desc_filename)

        # we first scale this model by number of cells, keeping reductions same as in search
        n_cells = math.ceil(len(template_model_desc.cell_descs())*cell_count_scale)
        n_cells = max(n_cells, max_cells)

        conf_model_desc = copy.deepcopy(conf_model_desc)
        conf_model_desc['n_cells'] = n_cells
        conf_model_desc['n_reductions'] = template_model_desc.cell_type_count(CellType.Reduction)

        model_desc = model_desc_builder.build(conf_model_desc,
                                              template=template_model_desc)
        # save desc for reference
        model_desc.save(full_desc_filename)

        model = evaluater.model_from_desc(model_desc)

        train_metrics = evaluater.train_model(conf_eval, model, checkpoint)
        train_metrics.save(metrics_filename)

        # get metrics_stats
        model_stats = nas_utils.get_model_stats(model)
        # save metrics_stats
        with open(model_stats_filename, 'w') as f:
            yaml.dump(model_stats, f)

        # save model
        if model_filename:
            model_filename = utils.full_path(model_filename)
            ml_utils.save_model(model, model_filename)
            logger.info({'model_save_path': model_filename})

        if checkpoint:
            checkpoint.new()
            checkpoint['metrics_stats'] = train_metrics, model_stats
            checkpoint.commit()

        return train_metrics, model_stats


    def _plot_model_gallery(self, metric_stats_all: List[Tuple[Metrics, tw.ModelStats]])->None:
        assert(len(metric_stats_all) > 0)

        xs_madd = []
        xs_flops = []
        ys = []
        for metrics, model_stats in metric_stats_all:
            xs_madd.append(model_stats.MAdd)
            xs_flops.append(model_stats.Flops)
            ys.append(metrics.best_val_top1())

        expdir = get_expdir()
        assert expdir
        madds_plot_filename = os.path.join(expdir, 'model_gallery_accuracy_madds.png')

        plt.clf()
        plt.scatter(xs_madd, ys)
        plt.xlabel('Multiply-Additions')
        plt.ylabel('Top1 Accuracy')
        plt.savefig(madds_plot_filename, dpi=plt.gcf().dpi, bbox_inches='tight')

        flops_plot_filename = os.path.join(expdir, 'model_gallery_accuracy_flops.png')

        plt.clf()
        plt.scatter(xs_flops, ys)
        plt.xlabel('Flops')
        plt.ylabel('Top1 Accuracy')
        plt.savefig(flops_plot_filename, dpi=plt.gcf().dpi, bbox_inches='tight')

    def _ensure_dataset_download(self, conf_search:Config)->None:
        conf_loader = conf_search['loader']
        self.get_data(conf_loader)


