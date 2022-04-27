from overrides import overrides
from typing import Optional, Type, Tuple

from archai.nas.exp_runner import ExperimentRunner
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
from archai.algos.evolution_pareto_image_seg.evolution_pareto_search_segmentation import EvolutionParetoSearchSegmentation



class EvolutionParetoSearchSegmentationEvaluater(Evaluater):
    ''' Evaluates a Segmentation Arch.'''
