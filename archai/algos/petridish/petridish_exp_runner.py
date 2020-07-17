# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Type, Optional, Tuple
import os
import shutil

from overrides import overrides

from archai.nas.exp_runner import ExperimentRunner
from archai.nas.arch_trainer import ArchTrainer, TArchTrainer
from archai.nas.search_distributed import SearchDistributed
from archai.nas import evaluate
from archai.common.config import Config
from archai.common import utils
from .petridish_cell_builder import PetridishCellBuilder

class PetridishExperimentRunner(ExperimentRunner):
    @overrides
    def cell_builder(self)->PetridishCellBuilder:
        return PetridishCellBuilder()

    @overrides
    def trainer_class(self)->TArchTrainer:
        return ArchTrainer


    @overrides
    def _run_search(self, conf_search: Config) -> None:
        ''' Using special search class designed for petridish '''
        cell_builder = self.cell_builder()
        trainer_class = self.trainer_class()
        finalizers = self.finalizers()

        search = SearchDistributed(conf_search, cell_builder, trainer_class, finalizers)
        search.search_loop()

    @overrides
    def _run_eval(self, conf_eval:Config)->None:
        evaluate.eval_archs(conf_eval, cell_builder=self.cell_builder())

    @overrides
    def _copy_final_desc(self, search_conf)->Tuple[Config, Config]:
        # get folder of model gallery that search has produced
        search_desc_foldername = search_conf['nas']['search']['final_desc_foldername']
        search_desc_folderpath = utils.full_path(search_desc_foldername)
        assert search_desc_foldername and os.path.exists(search_desc_folderpath)

        # get folder path that eval would need
        eval_conf = self._init('eval')
        eval_desc_foldername = eval_conf['nas']['eval']['final_desc_foldername']
        eval_desc_folderpath = utils.full_path(eval_desc_foldername)
        assert eval_desc_folderpath
        # only later version of shutil copytree has dirs_exists_ok option
        # so being robust to pre-existing directory
        if os.path.exists(eval_desc_folderpath):
            shutil.rmtree(eval_desc_folderpath)
        shutil.copytree(search_desc_folderpath, eval_desc_folderpath)

        return search_conf, eval_conf
    

