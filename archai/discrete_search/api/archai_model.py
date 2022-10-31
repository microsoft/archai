
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Dict, Any

class ArchaiModel():
    """Wraps a model object with an architecture id and optionally a metadata dictionary.

        Args:
            arch (Any): Model object (e.g torch.nn.Module)
            
            archid (str): **Architecture** string identifier of `arch` object. Will be used to deduplicate
                models of the same architecture, so architecture hashes are prefered. `archid` should only 
                identify neural network architectures and not model weight information.
            
            metadata (Optional[Dict], optional): Optional model metadata dictionary. Defaults to None.
        """

    def __init__(self, arch: Any, archid: str, metadata: Optional[Dict] = None):
        self.arch = arch
        self.archid = archid
        self.metadata = metadata or dict()

    def __repr__(self):
        return (
            f'ArchaiModel(\n\tarchid={self.archid}, \n\t'
            f'metadata={self.metadata}, \n\tarch={self.arch}\n)'
        )

    def __str__(self):
        return repr(self)

# from abc import abstractmethod
# from overrides import EnforceOverrides


# class ArchaiModel(EnforceOverrides):
  
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

