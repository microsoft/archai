from abc import abstractmethod
from typing import List
import logging

import torch

from overrides import overrides, EnforceOverrides

class VocabBase(EnforceOverrides):
    @abstractmethod
    def train(self, filepaths:List[str])->None:
        pass

    @abstractmethod
    def load(self)->None:
        pass

    @abstractmethod
    def encode_line(self, line:str)->List[int]:
        pass

    @abstractmethod
    def is_trained(self)->bool:
        pass

    @abstractmethod
    def __len__(self):
        pass

    def encode_file(self, path:str, verbose=True)->List[int]:
        logging.info(f'Encoding file: {path}')
        encoded = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    logging.info(f'    completed file line {format(idx)}')
                tokens = self.encode_line(line)
                encoded.extend(tokens)

        return encoded