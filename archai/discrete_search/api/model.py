
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Dict, Any

class NasModel():
    def __init__(self, arch: Any, archid: str, metadata: Optional[Dict] = None):
        """Used to wrap a NAS model.

        Args:
            arch (Any): Model object (e.g torch.nn.Module)
            archid (str): **Architecture** string identifier of `arch` object. Will be used to deduplicate
                models of the same architecture, so architecture hashes are prefered. `archid` should only 
                identify neural network architectures and not model weight information.
            metadata (Optional[Dict], optional): Optional model metadata dictionary. Defaults to None.
        """
        assert isinstance(archid, str)
        assert isinstance(metadata, dict)

        self.arch = arch
        self.archid = archid
        self.metadata = metadata


# from abc import abstractmethod
# from overrides import EnforceOverrides


# class NasModel(EnforceOverrides):
  
#     @property
#     @abstractmethod
#     def archid(self) -> str:
#         ''' Returns the model architecture identifier. ''' 

#     @abstractmethod
#     def load_weights(self, file: str) -> None:
#         ''' Loads model weights from `file`  ''' 

#     @abstractmethod
#     def save_weights(self, file: str) -> None:
#         ''' Saves current model weights to `file` ''' 

