# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import opacus

def _grad_sampler_get_attr(self, item):
    try:
        return super(opacus.GradSampleModule, self).__getattr__(item)
    except AttributeError as e:
        submodules = dict(self._module.named_modules())
        if item and item in submodules:
            return submodules[item]

        # Enable the Opacus wrapped model to access internal/ArchaiModel
        # model attributes
        if hasattr(self._module, item):
            return getattr(self._module, item)
        
        raise e

opacus.GradSampleModule.__getattr__ = _grad_sampler_get_attr