# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from overrides import EnforceOverrides
from typing import Any


class Searcher(EnforceOverrides):

    @abstractmethod
    def search(self) -> 'SearchResults':
        pass
