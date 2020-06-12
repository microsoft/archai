# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import UserDict
from typing import Dict, Iterable, Iterator, Mapping, Optional, Tuple, Union
import torch
from torch import nn

_param_suffix = '_arch_param' # all arch parameter names must have this suffix

NNTypes = Union[nn.Parameter, nn.ParameterDict, nn.ParameterList]

class ArchParams(UserDict):
    """This class holds set of learnable architecture parameter(s) for a given module. For example, one instance of this class would hold alphas for one instance of MixedOp. For sharing parameters, instance of this class can be passed around. Different algorithms may add learnable parameters for their need."""

    def __init__(self, arch_params:Iterable[Tuple[str, NNTypes]], registrar:Optional[nn.Module]=None):
        """Create architecture parameters and register them

        Arguments:
            registrar {Optional[nn.Module]} -- If this parameter is beingly newly created instead of being shared by other module then owner should be specified. When owner is not None, this method will create a variable in the owning module with suffix _arch_param so that the parameter gets registered with Pytorch and becomes available in module's .parameters() calls.
        """
        super().__init__()

        for name, param in arch_params:
            self.data[name] = param
            if registrar is not None:
                setattr(registrar, name + _param_suffix, param)

    def __setitem__(self, name:str, param:NNTypes)->None:
        raise RuntimeError(f'ArchParams is immutable hence adding/updating key {name} is not allowed.')

    def __delitem__(self, name:str) -> None:
        raise RuntimeError(f'ArchParams is immutable hence removing key {name} is not allowed.')

    def _by_kind(self, kind:Optional[str])->Iterator[NNTypes]:
        # TODO: may be optimize to avoid split() calls?
        for name, param in self.items():
            if kind is None or name.split('.')[-1]==kind:
                yield param

    def param_by_kind(self, kind:Optional[str])->Iterator[nn.Parameter]:
        # TODO: enforce type checking if debugger is active?
        return self._by_kind(kind) # type: ignore

    def paramlist_by_kind(self, kind:Optional[str])->Iterator[nn.ParameterList]:
        # TODO: enforce type checking if debugger is active?
        return self._by_kind(kind) # type: ignore

    def paramdict_by_kind(self, kind:Optional[str])->Iterator[nn.ParameterDict]:
        # TODO: enforce type checking if debugger is active?
        return self._by_kind(kind) # type: ignore

    def has_kind(self, kind:str)->bool:
        # TODO: may be optimize to avoid split() calls?
        for name in self.keys():
            if name.split('.')[-1]==kind:
                return True
        return False

    @staticmethod
    def from_module(module:nn.Module, recurse:bool=False)->'ArchParams':
        suffix_len = len(_param_suffix)
        # Pytorch named params have . in name for each module, we pick last part and remove _arch_params prefix
        arch_params = ((name[:-suffix_len], param) \
                       for name, param in module.named_parameters(recurse=recurse)
                       if name.endswith(_param_suffix))
        return ArchParams(arch_params)

    @staticmethod
    def nonarch_from_module(module:nn.Module, recurse:bool=False)->Iterator[nn.Parameter]:
        # Pytorch named params have . in name for each module, we pick last part and remove _arch_params prefix
        return (param for name, param in module.named_parameters(recurse=recurse)
                    if not name.endswith(_param_suffix))

    @staticmethod
    def empty()->'ArchParams':
        return ArchParams([])

