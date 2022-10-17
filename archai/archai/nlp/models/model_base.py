# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Model-related base class.
"""

from typing import Dict, List, Optional

import torch


def _get_layers_from_module(module: torch.nn.Module,
                            layer_type: Optional[str] = None) -> List[torch.nn.Module]:
    sub_module = list(module.children())
    layers = []

    if layer_type is not None:
        for lt in layer_type:
            if module.__class__.__name__ == lt:
                return module
    else:
        if len(sub_module) == 0 and len(list(module.parameters())) > 0:
            return module

    for m in sub_module:
        try:
            layers.extend(_get_layers_from_module(m, layer_type))
        except TypeError:
            layers.append(_get_layers_from_module(m, layer_type))

    return layers


class ArchaiModel(torch.nn.Module):
    """Base model class, used to define some common attributes
        and shared methods for standardizing inputs and outputs.

    """

    def __init__(self) -> None:
        """Initializes the class by overriding the standard torch.nn.Module.

        Note that ArchaiModels serves as an entrypoint for standardizing inputs
            and outputs, which are required by the provided scripts.
            
        """

        super().__init__()

    def get_params_from_layer(self, layer_type: str) -> int:
        layers = _get_layers_from_module(self, layer_type)
        n_params = {}

        for i, layer in enumerate(layers):
            layer_name = layer.__class__.__name__ + '_' + str(i)
            n_params[layer_name] = sum([p.nelement() for p in layer.parameters()])
        
        return sum(list(n_params.values()))

    def get_params(self) -> Dict[str, int]:
        params = {}

        params['total'] = 0
        params['non_embedding'] = 0

        return params

    def reset_length(self, tgt_len: int, ext_len: int, mem_len: int) -> None:
        return
