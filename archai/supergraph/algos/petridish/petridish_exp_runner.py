# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import shutil

from overrides import overrides

from archai.common import utils
from archai.supergraph.algos.petridish.evaluater_petridish import EvaluaterPetridish
from archai.supergraph.algos.petridish.petridish_model_desc_builder import (
    PetridishModelBuilder,
)
from archai.supergraph.algos.petridish.searcher_petridish import SearcherPetridish
from archai.supergraph.nas.arch_trainer import ArchTrainer, TArchTrainer
from archai.supergraph.nas.exp_runner import ExperimentRunner


def copy_dir(src_dir:str, dest_dir:str, use_shutil:bool=True)->None:
    if os.path.isdir(src_dir):
        if use_shutil:
            shutil.copytree(src_dir, dest_dir)
        else:
            if not os.path.isdir(dest_dir):
                os.makedirs(dest_dir)
            files = os.listdir(src_dir)
            for f in files:
                copy_dir(os.path.join(src_dir, f),
                        os.path.join(dest_dir, f), use_shutil=use_shutil)
    else:
        utils.copy_file(src_dir, dest_dir, use_shutil=use_shutil)


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




