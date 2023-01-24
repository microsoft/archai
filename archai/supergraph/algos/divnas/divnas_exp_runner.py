# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from overrides import overrides

from archai.supergraph.utils.common import get_conf
from archai.supergraph.nas.exp_runner import ExperimentRunner
from archai.supergraph.nas.arch_trainer import ArchTrainer, TArchTrainer
from archai.supergraph.algos.darts.bilevel_arch_trainer import BilevelArchTrainer
from archai.supergraph.algos.divnas.divnas_model_desc_builder import DivnasModelDescBuilder
from archai.supergraph.algos.divnas.divnas_finalizers import DivnasFinalizers
from archai.supergraph.algos.divnas.divnas_rank_finalizer import DivnasRankFinalizers
from archai.supergraph.nas.finalizers import Finalizers

class DivnasExperimentRunner(ExperimentRunner):

    @overrides
    def model_desc_builder(self)->DivnasModelDescBuilder:
        return DivnasModelDescBuilder()

    @overrides
    def trainer_class(self)->TArchTrainer:
        conf = get_conf()
        trainer = conf['nas']['search']['divnas']['archtrainer']

        if trainer == 'bilevel':
            return BilevelArchTrainer
        elif trainer == 'noalpha':
            return ArchTrainer
        else:
            raise NotImplementedError


    @overrides
    def finalizers(self)->Finalizers:
        conf = get_conf()
        finalizer = conf['nas']['search']['finalizer']

        if finalizer == 'mi':
            return DivnasFinalizers()
        elif finalizer == 'mi_ranked':
            return DivnasRankFinalizers()
        else:
            return super().finalizers()



