# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Base model class, used to defined some common functionalities.
"""

import logging
import os
from abc import abstractmethod
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn


class ArchaiModel(nn.Module):
    """Base model that abstracts further models definitions.
    
    """

    @abstractmethod
    def reset_length(self,
                     tgt_len: int,
                     ext_len: int,
                     mem_len: int) -> None:
        """Resets the length of the memory (used by Transformer-XL).

        Args:
            tgt_len: Length of target sample.
            ext_len: Length of extended memory.
            mem_len: Length of the memory.

        """

        pass

    @abstractmethod
    def get_non_emb_params(self) -> int:
        """Returns the number of non-embedding parameters.

        Returns:
            (int): Number of non-embedding parameters.

        """

        pass

    def get_n_params(self) -> int:
        """Returns the number of total parameters.

        Returns:
            (int): Number of total parameters.
            
        """
        
        return sum([p.nelement() for p in self.parameters()])

    def update_with_checkpoint(self,
                               checkpoint_folder_path,
                               on_cpu=False,
                               for_export=False):
        """Updates current model with a previously trained checkpoint.

        Args:

        Returns:

        """

        if os.path.isdir(checkpoint_folder_path):
            checkpoint_path = os.path.join(checkpoint_folder_path, 'checkpoint_last.pt')

        device = f'cuda:{torch.cuda.current_device()}' if not on_cpu and torch.cuda.is_available() else torch.device('cpu')

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_config = checkpoint['model_config']

        if for_export:
            model_config['use_cache'] = True

        self.load_state_dict(checkpoint['model_state'])

        return model_config
