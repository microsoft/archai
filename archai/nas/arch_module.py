# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC
from typing import Iterable, List, Optional, Tuple, Iterator

from torch import nn

from overrides import overrides, EnforceOverrides

from archai.nas.arch_params import ArchParams, NNTypes
from archai.common import utils

class ArchModule(nn.Module, ABC, EnforceOverrides):
    """ArchModule enahnces nn.Module by making a clear separation between regular
    weights and the architecture weights. The architecture parameters can be added
    using  `create_arch_params()` method and then accessed using `arch_params()` method."""

    def __init__(self) -> None:
        super().__init__()

        # these are params module should use, they may be shared or created by this module
        self._arch_params = ArchParams.empty()
        # these are the params created and registerd in this module
        self._owned_arch_params:Optional[ArchParams] = None

    def create_arch_params(self, named_params:Iterable[Tuple[str, NNTypes]])->None:
        if len(self._arch_params):
            raise RuntimeError('Arch parameters for this module already exist')
        self._owned_arch_params = ArchParams(named_params, registrar=self)
        self.set_arch_params(self._owned_arch_params)

    def set_arch_params(self, arch_params:ArchParams)->None:
        if len(self._arch_params):
            raise RuntimeError('Arch parameters for this module already exist')
        self._arch_params = arch_params

    def arch_params(self, recurse=False, only_owned=False)->ArchParams:
        # note that we will cache lists on first calls, this doesn't allow
        # dynamic parameters but it makes this frequent calls much faster
        if not recurse:
            if not only_owned:
                return self._arch_params
            else:
                return ArchParams.from_module(self, recurse=False)
        else:
            if not only_owned:
                raise NotImplementedError('Recursively getting shared and owned arch params not implemented yet')
            else:
                return ArchParams.from_module(self, recurse=True)

    def all_owned(self)->ArchParams:
        return self.arch_params(recurse=True, only_owned=True)

    def nonarch_params(self, recurse:bool)->Iterator[nn.Parameter]:
        return ArchParams.nonarch_from_module(self, recurse)