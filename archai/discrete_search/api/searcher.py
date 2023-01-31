# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod

from overrides import EnforceOverrides

from archai.discrete_search.api.search_results import SearchResults


class Searcher(EnforceOverrides):
    """Abstract class for searchers.

    This class serves as a base for implementing searchers, which searches for an
    architecture given an algorithm. The class enforces implementation of a single
    method: `search`.

    Note:
        This class is inherited from `EnforceOverrides` and any overridden methods in the
        subclass should be decorated with `@overrides` to ensure they are properly overridden.

    Examples:
        >>> class MySearcher(Searcher):
        >>>     def __init__(self) -> None:
        >>>         super().__init__()
        >>>
        >>>     @overrides
        >>>     def search(self) -> SearchResults:
        >>>         # Code used to search for the best architecture
        >>>         return SearchResults(...)

    """

    def __init__(self) -> None:
        """Initialize the searcher."""

        pass

    @abstractmethod
    def search(self) -> SearchResults:
        """Search for the best architecture.

        Returns:
            Search results.

        """

        pass
