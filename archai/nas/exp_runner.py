# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Type, Tuple
from abc import ABC, abstractmethod
import shutil
import os

from overrides import EnforceOverrides

from archai.nas.model_desc_builder import ModelDescBuilder
from archai.nas.arch_trainer import TArchTrainer
from archai.common.common import common_init
from archai.common import utils
from archai.common.config import Config
from archai.nas.evaluater import Evaluater, EvalResult
from archai.nas.searcher import Searcher, SearchResult
from archai.nas.finalizers import Finalizers
from archai.common.common import get_conf
from archai.nas.random_finalizers import RandomFinalizers
from archai.nas.model_desc_builder import ModelDescBuilder


class ExperimentRunner(ABC, EnforceOverrides):
    def __init__(self, config_filename:str, base_name:str, clean_expdir=False) -> None:
        self.config_filename = config_filename
        self.base_name = base_name
        self.clean_expdir = clean_expdir

    def run_search(self, conf_search:Config)->SearchResult:
        model_desc_builder = self.model_desc_builder()
        trainer_class = self.trainer_class()
        finalizers = self.finalizers()

        search = self.searcher()
        return search.search(conf_search, model_desc_builder, trainer_class, finalizers)

    def run_eval(self, conf_eval:Config)->EvalResult:
        evaler = self.evaluater()
        return evaler.evaluate(conf_eval,
                               model_desc_builder=self.model_desc_builder())

    def run(self, search=True, eval=True) \
            ->Tuple[Optional[SearchResult], Optional[EvalResult]]:

        search_result, eval_result = None, None

        if search: # run search
            search_result = self.run_search(self.get_conf_search(clean_expdir=self.clean_expdir))

        if eval:
            if search:
                # below is only done to remove existing expdir
                _ = self.get_conf_eval(clean_expdir=self.clean_expdir)
                self.copy_search_to_eval()

            eval_result = self.run_eval(self.get_conf_eval())

        return search_result, eval_result

    def copy_search_to_eval(self)->None:
        # do not cache conf_search or conf_eval as it may have values that
        # needs env var expansion.

        # get desc file path that search has produced
        conf_search = self.get_conf_search()
        search_desc_filename = conf_search['final_desc_filename']
        search_desc_filepath = utils.full_path(search_desc_filename)
        assert search_desc_filepath and os.path.exists(search_desc_filepath)

        # get file path that eval would need
        conf_eval = self.get_conf_eval()
        eval_desc_filename = conf_eval['final_desc_filename']
        eval_desc_filepath = utils.full_path(eval_desc_filename)
        assert eval_desc_filepath
        utils.copy_file(search_desc_filepath, eval_desc_filepath)

    def model_desc_builder(self)->Optional[ModelDescBuilder]:
        return ModelDescBuilder() # default model desc builder puts nodes with no edges

    def searcher(self)->Searcher:
        return Searcher()

    def evaluater(self)->Evaluater:
        return Evaluater()

    @abstractmethod
    def trainer_class(self)->TArchTrainer:
        pass

    def finalizers(self)->Finalizers:
        conf = get_conf()
        finalizer = conf['nas']['search']['finalizer']

        if not finalizer or finalizer == 'default':
            return Finalizers()
        elif finalizer == 'random':
            return RandomFinalizers()
        else:
            raise NotImplementedError

    def get_conf_search(self, clean_expdir=False)->Config:
        conf = self._init_conf('search', clean_expdir)
        conf_search = conf['nas']['search']
        return conf_search

    def get_conf_eval(self, clean_expdir=False)->Config:
        conf = self._init_conf('eval', clean_expdir)
        conf_eval = conf['nas']['eval']
        return conf_eval

    def _init_conf(self, exp_name_suffix:str, clean_expdir:bool)->Config:
        config_filename = self.config_filename

        conf = common_init(config_filepath=config_filename,
            param_args=['--common.experiment_name', self.base_name + f'_{exp_name_suffix}',
                        ], clean_expdir=clean_expdir)
        return conf

