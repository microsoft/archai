
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Dict, Any

class NasModel():
    def __init__(self, arch: Any, archid: str, metadata: Optional[Dict] = None):
        """Neural Architecture Search model.

        Args:
            arch (Any): Callable model object
            archid (str): Architecture identifier.
            metadata (Optional[Dict], optional): Extra model metadata. Defaults to None.
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

