from abc import abstractmethod
from typing import List, Optional
import logging
from collections import abc


from overrides import overrides, EnforceOverrides
import torch

from archai.nlp.datasets.tokenizer_utils.special_token_enum import SpecialTokenEnum

class VocabBase(EnforceOverrides, abc.Sized):
    @abstractmethod
    def train(self, filepaths:List[str])->None:
        pass

    @abstractmethod
    def load(self)->None:
        pass

    @abstractmethod
    def encode_text(self, text:str)->List[int]:
        pass
    @abstractmethod
    def decode_text(self, ids:List[int])->str:
        pass

    @abstractmethod
    def is_trained(self)->bool:
        pass

    @abstractmethod
    @overrides
    def __len__(self):
        pass

    @abstractmethod
    def token_to_id(self, t:str)->int:
        pass
    @abstractmethod
    def id_to_token(self, id:int)->str:
        pass

    @abstractmethod
    def special_token_id(self, sp:SpecialTokenEnum)->Optional[int]:
        pass

    def tokens_to_ids(self, ts:List[str])->List[int]:
        return [self.token_to_id(t) for t in ts]
    def ids_to_tokens(self, ids:List[int])->List[str]:
        return [self.id_to_token(id) for id in ids]

    def encode_file(self, path:str, verbose=True)->List[int]:
        logging.info(f'Encoding file: {path}')
        encoded = []
        tensor_encoded = torch.LongTensor()

        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):

                # Converts to tensor.Tensor every 500k lines otherwise Python list uses a lot of RAM
                if idx > 0 and idx % 500000 == 0:
                    tensor_encoded = torch.cat((tensor_encoded, torch.LongTensor(encoded)))
                    encoded = []

                if verbose and idx > 0 and idx % 500000 == 0:
                    logging.info(f'    completed file line {format(idx)}')

                tokens = self.encode_text(line)
                encoded.extend(tokens)

        if len(encoded) > 0:
            tensor_encoded = torch.cat((tensor_encoded, torch.LongTensor(encoded)))

        return tensor_encoded