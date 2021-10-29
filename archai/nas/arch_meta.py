from typing import Dict
import torch.nn as nn


class ArchWithMetaData:
    def __init__(self, model:nn.Module, extradata:Dict):
        self._arch = model
        self._metadata = extradata

    @property
    def arch(self):
        return self._arch

    @arch.setter
    def arch(self, model:nn.Module):
        assert isinstance(model, nn.Module)
        self._arch = model

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, extradata:Dict):
        assert isinstance(extradata, dict) 
        self._metadata = extradata
    