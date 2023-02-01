# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from overrides import overrides

from archai.discrete_search.api.search_results import SearchResults
from archai.discrete_search.api.searcher import Searcher


class MySearcher(Searcher):
    def __init__(self) -> None:
        super().__init__()

    @overrides
    def search(self) -> SearchResults:
        return SearchResults(None, None)


def test_searcher():
    searcher = MySearcher()

    # Assert that mocked method return a `SearchResults`
    search_results = searcher.search()
    assert isinstance(search_results, SearchResults)
