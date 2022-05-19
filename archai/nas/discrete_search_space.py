# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABCMeta, abstractmethod
from typing import List, Union
from overrides.enforce import EnforceOverrides

import torch
import torch_geometric
from archai.nas.arch_meta import ArchWithMetaData


class DiscreteSearchSpace(EnforceOverrides, metaclass=ABCMeta):
    @abstractmethod
    def random_sample(self)->ArchWithMetaData:
        '''Uniform random sample an architecture (nn.Module)'''
        pass

    @abstractmethod
    def get_neighbors(self, arch:ArchWithMetaData)->List[ArchWithMetaData]:
        '''Return the neighbors (some definition of neighborhood) of an architecture'''
        pass

class EncodableDiscreteSearchSpace(DiscreteSearchSpace):
    @abstractmethod
    def get_arch_repr(self, arch:ArchWithMetaData)->Union[torch.Tensor, torch_geometric.data.Data]:
        """Gets a representation (either a torch.Tensor or a torch_geometric.data.Data graph)
        of an architecture from the search space.

        Args:
            arch (ArchWithMetaData): An architecture from the search space.

        Returns:
            Union[torch.Tensor, Data]: A representation of the architecture.
        """        
        pass
