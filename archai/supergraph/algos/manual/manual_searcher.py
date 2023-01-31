# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

from overrides import overrides

from archai.common.config import Config
from archai.supergraph.nas.arch_trainer import TArchTrainer
from archai.supergraph.nas.finalizers import Finalizers
from archai.supergraph.nas.model_desc_builder import ModelDescBuilder
from archai.supergraph.nas.searcher import Searcher, SearchResult


class ManualSearcher(Searcher):
    @overrides
    def search(self, conf_search:Config, model_desc_builder:Optional[ModelDescBuilder],
                 trainer_class:TArchTrainer, finalizers:Finalizers)->SearchResult:
        # for manual search, we already have a model so no search result are returned
        return SearchResult(None, None, None)