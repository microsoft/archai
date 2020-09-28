# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Type, Optional, Tuple
import os
import shutil

from overrides import overrides

from archai.nas.exp_runner import ExperimentRunner
from archai.nas.arch_trainer import ArchTrainer, TArchTrainer
from .petridish_model_desc_builder import PetridishModelBuilder

from .searcher_petridish import SearcherPetridish
from .evaluater_petridish import EvaluaterPetridish
from archai.common.config import Config
from archai.common import utils


class PetridishExperimentRunner(ExperimentRunner):
    @overrides
    def model_desc_builder(self)->PetridishModelBuilder:
        return PetridishModelBuilder()

    @overrides
    def trainer_class(self)->TArchTrainer:
        return ArchTrainer

    @overrides
    def searcher(self)->SearcherPetridish:
        return SearcherPetridish()

    @overrides
    def evaluater(self)->EvaluaterPetridish:
        return EvaluaterPetridish()

    @overrides
    def copy_search_to_eval(self)->None:
        # get folder of model gallery that search has produced
        conf_search = self.get_conf(True)['nas']['search']
        search_desc_foldername = conf_search['final_desc_foldername']
        search_desc_folderpath = utils.full_path(search_desc_foldername)
        assert search_desc_foldername and os.path.exists(search_desc_folderpath)

        # get folder path that eval would need
        conf_eval = self.get_conf(False)['nas']['eval']
        eval_desc_foldername = conf_eval['final_desc_foldername']
        eval_desc_folderpath = utils.full_path(eval_desc_foldername)
        assert eval_desc_folderpath
        # only later version of shutil copytree has dirs_exists_ok option
        # so being robust to pre-existing directory
        if os.path.exists(eval_desc_folderpath):
            shutil.rmtree(eval_desc_folderpath)
        utils.copy_dir(search_desc_folderpath, eval_desc_folderpath, use_shutil=True)




