# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Iterator, Mapping, Type, Optional, Tuple, List
import math
import copy
import random
import os

from overrides import overrides

from torch.utils.data.dataloader import DataLoader


from archai.common.common import logger

from archai.common.config import Config
from archai.supergraph.nas.model_desc_builder import ModelDescBuilder
from archai.supergraph.nas.arch_trainer import TArchTrainer
from archai.supergraph.utils.trainer import Trainer
from archai.supergraph.nas.model_desc import CellType, ModelDesc
from archai.supergraph.datasets import data
from archai.supergraph.nas.model import Model
from archai.supergraph.utils.metrics import EpochMetrics, Metrics
from archai.common import utils
from archai.supergraph.nas.finalizers import Finalizers
from archai.supergraph.nas.searcher import Searcher, SearchResult

class ManualSearcher(Searcher):
    @overrides
    def search(self, conf_search:Config, model_desc_builder:Optional[ModelDescBuilder],
                 trainer_class:TArchTrainer, finalizers:Finalizers)->SearchResult:
        # for manual search, we already have a model so no search result are returned
        return SearchResult(None, None, None)