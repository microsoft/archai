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
from archai.nas.model_desc_builder import ModelDescBuilder
from archai.nas.arch_trainer import TArchTrainer
from archai.common.trainer import Trainer
from archai.nas.model_desc import CellType, ModelDesc
from archai.datasets import data
from archai.nas.model import Model
from archai.common.metrics import EpochMetrics, Metrics
from archai.common import utils
from archai.nas.finalizers import Finalizers
from archai.nas.searcher import Searcher, SearchResult

class ManualSearcher(Searcher):
    @overrides
    def search(self, conf_search:Config, model_desc_builder:Optional[ModelDescBuilder],
                 trainer_class:TArchTrainer, finalizers:Finalizers)->SearchResult:
        # for manual search, we already have a model so no search result are returned
        return SearchResult(None, None, None)