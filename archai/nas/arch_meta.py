from typing import Dict
import torch.nn as nn


class ArchWithMetaData:
    def __init__(self, arch:nn.Module, metadata:Dict):
        assert isinstance(arch, nn.Module)
        assert isinstance(metadata, dict)
        self.arch = arch
        self.metadata = metadata

    @property
    def arch(self):
        return self.arch

    @arch.setter
    def arch(self, arch:nn.Module):
        assert isinstance(arch, nn.Module)
        self.arch = arch

    @property
    def metadata(self):
        return self.metadata

    @metadata.setter
    def metadata(self, metadata):
        assert isinstance(metadata, dict) 
        self.metadata = metadata
    