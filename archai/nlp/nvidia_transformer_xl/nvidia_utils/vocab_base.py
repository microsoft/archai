import logging
from abc import abstractmethod
import os
from collections import Counter
from collections import OrderedDict
from typing import List, Optional

import torch

from overrides import overrides, EnforceOverrides

class VocabBase(EnforceOverrides):
    @abstractmethod
    def train(self, filepaths:List[str], save_dir:str)->None:
        pass

    @abstractmethod
    def load(self, path:str)->bool:
        pass

    @abstractmethod
    def exists(self, path:str)->bool:
        pass

    def encode_file(self, path:str, verbose=True)->torch.Tensor:
        logging.info(f'Encoding file: {path}')
        encoded = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    logging.info(f'    completed file line {format(idx)}')
                tokens = self.encode_line(line)
                encoded.append(tokens)

        encoded = torch.cat(encoded)
        return encoded

    @abstractmethod
    def encode_line(self, line):
        pass

    @abstractmethod
    def __len__(self):
        pass

