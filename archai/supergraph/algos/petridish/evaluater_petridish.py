# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import glob
import math
import os
import pathlib
from typing import Optional

# only works on linux
import ray
import yaml
from overrides import overrides

from archai.common import common, ml_utils, utils
from archai.common.config import Config
from archai.common.ordered_dict_logger import get_global_logger
from archai.supergraph.algos.petridish.petridish_utils import (
    ConvexHullPoint,
    ExperimentStage,
    JobStage,
    plot_pool,
    save_hull,
)
from archai.supergraph.nas import nas_utils
from archai.supergraph.nas.evaluater import EvalResult, Evaluater
from archai.supergraph.nas.model_desc import CellType, ModelDesc
from archai.supergraph.nas.model_desc_builder import ModelDescBuilder

logger = get_global_logger()


def filepath_ext(filepath:str)->str:
    """Returns '.f' for '/a/b/c/d.e.f' """
    return pathlib.Path(filepath).suffix

def filepath_name_only(filepath:str)->str:
    """Returns 'd.e' for '/a/b/c/d.e.f' """
    return pathlib.Path(filepath).stem

def append_to_filename(filepath:str, name_suffix:str, new_ext:Optional[str]=None)->str:
    """Returns '/a/b/c/h.f' for filepath='/a/b/c/d.e.f', new_name='h' """
    ext = new_ext or filepath_ext(filepath)
    name = filepath_name_only(filepath)
    return str(pathlib.Path(filepath).with_name(name+name_suffix).with_suffix(ext))


class EvaluaterPetridish(Evaluater):

    @overrides
    def evaluate(self, conf_eval:Config, model_desc_builder:ModelDescBuilder)->EvalResult:
        """Takes a folder of model descriptions output by search process and
        trains them in a distributed manner using ray with 1 gpu"""

        logger.pushd('evaluate')

        final_desc_foldername:str = conf_eval['final_desc_foldername']
        source_desc_foldername:str = conf_eval.get('source_desc_foldername', final_desc_foldername)

        # get list of model descs in the gallery folder
        source_desc_folderpath = utils.full_path(source_desc_foldername)
        final_desc_folderpath = utils.full_path(final_desc_foldername, True)

        files = [os.path.join(final_desc_folderpath, utils.filepath_name_ext(f)) \
                for f in glob.glob(os.path.join(source_desc_folderpath, 'model_desc_*.yaml')) \
                    if os.path.isfile(os.path.join(source_desc_folderpath, f))]
        logger.info({'model_desc_files':len(files)})

        # to avoid all workers download datasets individually, let's do it before hand
        self._ensure_dataset_download(conf_eval)

        future_ids = []
        for model_desc_filename in files:
            future_id = EvaluaterPetridish._train_dist.remote(self, conf_eval, model_desc_builder, model_desc_filename, common.get_state(), source_desc_folderpath)
            future_ids.append(future_id)

        # wait for all eval jobs to be finished
        ready_refs, remaining_refs = ray.wait(future_ids, num_returns=len(future_ids))

        # plot pareto curve of gallery of models
        hull_points = [ray.get(ready_ref) for ready_ref in ready_refs]
        save_hull(hull_points, common.get_expdir())
        plot_pool(hull_points, common.get_expdir(), ExperimentStage.EVAL)

        best_point = max(hull_points, key=lambda p:p.metrics.best_val_top1())
        logger.info({'best_val_top1':best_point.metrics.best_val_top1(),
                     'best_MAdd': best_point.model_stats.MAdd})

        logger.popd()

        return EvalResult(best_point.metrics)

    @staticmethod
    @ray.remote(num_gpus=1)
    def _train_dist(evaluater:Evaluater, conf_eval:Config, model_desc_builder:ModelDescBuilder,
                    model_desc_filename:str, common_state, source_folder:Optional[str])->ConvexHullPoint:
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
            conf_checkpoint['filename'] = utils.append_to_filename(model_filename, '_checkpoint', '.pth')
            checkpoint = nas_utils.create_checkpoint(conf_checkpoint, resume)

            if checkpoint is not None and resume:
                if 'metrics_stats' in checkpoint:
                    # return the output we had recorded in the checkpoint
                    convex_hull_point = checkpoint['metrics_stats']
                    return convex_hull_point

        # template model is what we used during the search
        if source_folder:
            template_model_desc = ModelDesc.load(os.path.join(source_folder, utils.filepath_name_ext(model_desc_filename)))
        else:
            template_model_desc = ModelDesc.load(model_desc_filename)

        # we first scale this model by number of cells, keeping reductions same as in search
        n_cells = math.ceil(len(template_model_desc.cell_descs())*cell_count_scale)
        n_cells = min(n_cells, max_cells)

        conf_model_desc = copy.deepcopy(conf_model_desc)
        conf_model_desc['n_cells'] = n_cells
        conf_model_desc['n_reductions'] = n_reductions = template_model_desc.cell_type_count(CellType.Reduction)

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
            # TODO: Causes logging error at random times. Commenting out as stop-gap fix.
            # logger.info({'model_save_path': model_filename})

        hull_point = ConvexHullPoint(JobStage.EVAL_TRAINED, 0, 0, model_desc,
                        (n_cells, n_reductions, len(model_desc.cell_descs()[0].nodes())),
                        metrics=train_metrics, model_stats=model_stats)

        if checkpoint:
            checkpoint.new()
            checkpoint['metrics_stats'] = hull_point
            checkpoint.commit()

        return hull_point

    def _ensure_dataset_download(self, conf_search:Config)->None:
        conf_loader = conf_search['loader']
        self.get_data(conf_loader)

