# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod

from overrides import EnforceOverrides


class DatasetProvider(EnforceOverrides):
    """"""

    def __init__(self) -> None:
        """"""

        super().__init__()

    @abstractmethod
    def get_dataset():
        """"""

    @abstractmethod
    def get_transform():
        """"""
