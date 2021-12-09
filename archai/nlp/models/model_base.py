# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
from abc import abstractmethod
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn


class ArchaiModel(nn.Module):
    """Abstract model that is compatible with nvidia's transformer xl code base"""

    @abstractmethod
    def reset_length(self, tgt_len:int, ext_len:int, mem_len:int):
        pass

    @abstractmethod
    def get_non_emb_params(self):
        pass

    def get_n_params(self):
        return sum([p.nelement() for p in self.parameters()])

    @staticmethod
    def load_model(model_cls: Type['ArchaiModel'], path:str, model:Optional['ArchaiModel']=None, on_cpu:Optional[bool]=False, for_export:Optional[bool]=False) -> Tuple['ArchaiModel', dict, dict]:

        # case for restart
        if os.path.isdir(path):
            path = os.path.join(path, 'checkpoint_last.pt')

        logging.info(f'Loading {model.__class__.__name__} model from: {path}')

        device = f'cuda:{torch.cuda.current_device()}' if not on_cpu and torch.cuda.is_available() else torch.device('cpu')

        # Loads the checkpoint
        checkpoint = torch.load(path, map_location=device)
        model_config = checkpoint['model_config']

        if for_export:
            model_config['use_cache'] = True

        # Initializes the model
        model = model_cls(**model_config) if model is None else model
        model.load_state_dict(checkpoint['model_state'])

        return model, model_config, checkpoint
