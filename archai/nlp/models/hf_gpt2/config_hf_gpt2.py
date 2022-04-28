# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Huggingface's Open AI GPT-2 configurations.
"""

from typing import List, Optional, Union

from archai.nlp.models.config_base import Config, SearchConfigParameter, SearchConfig
from transformers import CONFIG_MAPPING


class HfGPT2Config(Config):
    """Huggingface's Open AI GPT-2 default configuration.

    """

    attribute_map = {
        'n_token': 'vocab_size',
        'tgt_len': 'n_positions',
        'd_model': 'n_embd',
        'd_inner': 'n_inner',
        'dropout': 'resid_pdrop',
        'dropatt': 'attn_pdrop',
        'weight_init_std': 'initializer_range'
    }
    attribute_map.update(CONFIG_MAPPING['gpt2']().attribute_map)

    def __init__(self,
                 n_token: Optional[int] = 10000, # changed from 50257 for model's production
                 tgt_len: Optional[int] = 192,
                 d_model: Optional[int] = 512,
                 d_inner: Optional[int] = 2048,
                 dropout: Optional[float] = 0.1,
                 dropatt: Optional[float] = 0.0,
                 weight_init_std: Optional[float] = 0.02,
                 n_layer: Optional[int] = 16,
                 n_head: Optional[int] = 8,
                 embd_pdrop: Optional[float] = 0.0,
                 tie_weight: Optional[bool] = True,
                 **kwargs) -> None:
        """Initializes the class by overriding default arguments.

        Args:
            n_token: Size of the vocabulary (number of tokens).
            tgt_len: Maximum length of sequences (positional embeddings).
            d_model: Dimensionality of the model (also for embedding layer).
            d_inner: Dimensionality of inner feed-forward layers.
            dropout: Dropout probability.
            dropatt: Attention dropout probability.
            weight_init_std: Standard deviation to initialize the weights.
            n_layer: Number of layers.
            n_head: Number of attention heads.
            embd_pdrop: Dropout probability of embedding layer.
            tie_weight: Whether embedding and softmax weights should be tied.

        """

        self.n_token = n_token
        self.tgt_len = tgt_len
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.dropatt = dropatt
        self.weight_init_std = weight_init_std
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.embd_pdrop = embd_pdrop
        self.tie_weight = tie_weight

        additional_config = CONFIG_MAPPING['gpt2']().to_dict()
        for key, value in additional_config.items():
            if key not in self.__dict__.keys():
                setattr(self, key, value)

        super().__init__(**kwargs)


class HfGPT2SearchConfig(SearchConfig):
    """Huggingface's Open AI GPT-2 search configuration.

    """

    def __init__(self) -> None:
        """Initializes the class by setting default parameters that are used during search.
        
        """
        
        # Default HfGPT2 search options: n_layer, d_model, d_inner and n_head
        n_layer = SearchConfigParameter(per_layer=False, value=[3, 4, 5, 6, 7, 8, 9, 10])
        d_model = SearchConfigParameter(per_layer=False, value=list(range(128, 1024, 64)))
        d_inner = SearchConfigParameter(per_layer=False, value=list(range(128, 4096, 64)))
        n_head = SearchConfigParameter(per_layer=False, value=[2, 4, 8])

        super().__init__(n_layer=n_layer,
                         d_model=d_model,
                         d_inner=d_inner,
                         n_head=n_head)


class HfGPT2FlexConfig(HfGPT2Config):
    """Huggingface's Open AI GPT-2 Flex (different parameters per layer) default configuration.

    """

    def __init__(self,
                 n_token: Optional[int] = 10000, # changed from 50257 for model's production
                 tgt_len: Optional[int] = 192,
                 d_model: Optional[int] = 512,
                 d_inner: Optional[Union[int, List[int]]] = 2048,
                 dropout: Optional[float] = 0.1,
                 dropatt: Optional[float] = 0.0,
                 weight_init_std: Optional[float] = 0.02,
                 n_layer: Optional[int] = 16,
                 n_head: Optional[Union[int, List[int]]] = 8,
                 embd_pdrop: Optional[float] = 0.0,
                 tie_weight: Optional[bool] = True,
                 primer_square: Optional[bool] = False,
                 **kwargs) -> None:
        """Initializes the class by overriding default arguments.

        Args:
            n_token: Size of the vocabulary (number of tokens).
            tgt_len: Maximum length of sequences (positional embeddings).
            d_model: Dimensionality of the model (also for embedding layer).
            d_inner: Dimensionality of inner feed-forward layers.
            dropout: Dropout probability.
            dropatt: Attention dropout probability.
            weight_init_std: Standard deviation to initialize the weights.
            n_layer: Number of layers.
            n_head: Number of attention heads.
            embd_pdrop: Dropout probability of embedding layer.
            tie_weight: Whether embedding and softmax weights should be tied.
            primer_square: Whether squared ReLU primitive should be employed.

        """

        super().__init__(n_token=n_token,
                         tgt_len=tgt_len,
                         d_model=d_model,
                         dropout=dropout,
                         dropatt=dropatt,
                         weight_init_std=weight_init_std,
                         n_layer=n_layer,
                         embd_pdrop=embd_pdrop,
                         tie_weight=tie_weight,
                         **kwargs)

        self.d_inner = self._map_to_list(d_inner, n_layer)
        self.n_head = self._map_to_list(n_head, n_layer)
        self.d_head = [d_model // n_h for n_h in self.n_head]

        self.primer_square = primer_square
        if primer_square:
            self.activation_function = 'relu'


class HfGPT2FlexSearchConfig(SearchConfig):
    """Huggingface's Open AI GPT-2 Flex (different parameters per layer) search configuration.

    """

    def __init__(self) -> None:
        """Initializes the class by setting default parameters that are used during search.

        """

        # Default HfGPT2Flex search options: n_layer, d_model, d_inner and n_head
        n_layer = SearchConfigParameter(per_layer=False, value=[3, 4, 5, 6, 7, 8, 9, 10])
        d_model = SearchConfigParameter(per_layer=False, value=list(range(128, 1024, 64)))
        d_inner = SearchConfigParameter(per_layer=True, value=list(range(128, 4096, 64)))
        n_head = SearchConfigParameter(per_layer=False, value=[2, 4, 8])

        super().__init__(n_layer=n_layer,
                         d_model=d_model,
                         d_inner=d_inner,
                         n_head=n_head)
