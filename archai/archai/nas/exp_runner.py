# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Type, Tuple
from abc import ABC, abstractmethod
import shutil
import os

from overrides import EnforceOverrides

from archai.nas.model_desc_builder import ModelDescBuilder
from archai.nas.arch_trainer import TArchTrainer
from archai.common import common
from archai.common import utils
from archai.common.config import Config
from archai.nas.evaluater import Evaluater, EvalResult
from archai.nas.searcher import Searcher, SearchResult
from archai.nas.finalizers import Finalizers
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
            conf = self._init_conf(True, clean_expdir=self.clean_expdir)
            search_result = self.run_search(conf['nas']['search'])

        if eval:
            conf = self.get_conf(False)
            common.clean_ensure_expdir(conf, clean_dir=self.clean_expdir, ensure_dir=True)
            if search:
                # first copy search result to eval, otherwise we expect eval config to point to results
                self.copy_search_to_eval()

            conf = self._init_conf(False, clean_expdir=False)
            eval_result = self.run_eval(conf['nas']['eval'])

        return search_result, eval_result

    def copy_search_to_eval(self)->None:
        # do not cache conf_search or conf_eval as it may have values that
        # needs env var expansion.

        # get desc file path that search has produced
        conf_search = self.get_conf(True)['nas']['search']
        search_desc_filename = conf_search['final_desc_filename']
        search_desc_filepath = utils.full_path(search_desc_filename)
        assert search_desc_filepath and os.path.exists(search_desc_filepath)

        # get file path that eval would need
        conf_eval = self.get_conf(False)['nas']['eval']
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
        conf = common.get_conf()
        finalizer = conf['nas']['search']['finalizer']

        if not finalizer or finalizer == 'default':
            return Finalizers()
        elif finalizer == 'random':
            return RandomFinalizers()
        else:
            raise NotImplementedError

    def get_expname(self, is_search_or_eval:bool)->str:
        return self.base_name + ('_search' if is_search_or_eval else '_eval')

    def get_conf(self, is_search_or_eval:bool)->Config:
        conf = common.create_conf(config_filepath=self.config_filename,
            param_args=['--common.experiment_name', self.get_expname(is_search_or_eval)])
        common.update_envvars(conf) # config paths might include env vars
        return conf

    def _init_conf(self, is_search_or_eval:bool, clean_expdir:bool)->Config:
        config_filename = self.config_filename

        conf = common.common_init(config_filepath=config_filename,
            param_args=['--common.experiment_name', self.get_expname(is_search_or_eval),
                        ], clean_expdir=clean_expdir)
        return conf

