# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Base model class, used to defined some common functionalities.
"""

from abc import abstractmethod

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
