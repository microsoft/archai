# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Base model class, used to defined some common functionalities.
"""

import torch.nn as nn


class ArchaiModel(nn.Module):
    """Base model that abstracts further models definitions.
    
    """

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

        raise NotImplementedError

    def get_non_emb_params(self) -> int:
        """Returns the number of non-embedding parameters.

        Returns:
            (int): Number of non-embedding parameters.

        """

        raise NotImplementedError

    def get_n_params(self) -> int:
        """Returns the number of total parameters.

        Returns:
            (int): Number of total parameters.
            
        """
        
        return sum([p.nelement() for p in self.parameters()])
