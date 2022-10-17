# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from overrides import EnforceOverrides


class DatasetProvider(EnforceOverrides):
    @abstractmethod
    def get_train_val_datasets(self, *args, **kwargs):
        pass
