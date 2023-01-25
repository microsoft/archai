# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from overrides import EnforceOverrides

from archai.api.search_objectives import SearchObjectives
from archai.api.search_space import SearchSpace


class SearchResults(EnforceOverrides):
    """Abstract class for search results.

    This class serves as a base for implementing search results, which consists
    in producing data frames and plots with information regarding the search.

    Note:
        This class is inherited from `EnforceOverrides` and any overridden methods in the
        subclass should be decorated with `@overrides` to ensure they are properly overridden.

    """

    def __init__(self, search_space: SearchSpace, objectives: SearchObjectives) -> None:
        """Initialize the search results.

        Args:
            search_space: Search space.
            objectives: Search objectives.

        """

        self.search_space = search_space
        self.objectives = objectives
