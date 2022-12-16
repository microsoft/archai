# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from overrides import EnforceOverrides

from archai.discrete_search.api.search_results import SearchResults


class Searcher(EnforceOverrides):

    @abstractmethod
    def search(self) -> SearchResults:
        pass
