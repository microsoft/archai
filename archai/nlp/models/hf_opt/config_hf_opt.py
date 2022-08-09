# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Huggingface's Open Pre-Trained Transformer configurations.
"""

from typing import Optional

from archai.nlp.models.config_base import Config, SearchConfigParameter, SearchConfig
from transformers import CONFIG_MAPPING


class HfOPTConfig(Config):
    """Huggingface's Open Pre-Trained Transformer default configuration.

    """

    attribute_map = {
        'n_token': 'vocab_size',
        'tgt_len': 'max_position_embeddings',
        'd_model': 'hidden_size',
        'd_inner': 'ffn_dim',
        'n_layer': 'num_hidden_layers',
        'n_head': 'num_attention_heads',
        'dropatt': 'attention_dropout',
        'weight_init_std': 'init_std'
    }
    attribute_map.update(CONFIG_MAPPING['opt']().attribute_map)

    def __init__(self,
                 n_token: Optional[int] = 50272,
                 tgt_len: Optional[int] = 2048,
                 d_model: Optional[int] = 1024,
                 d_inner: Optional[int] = 4096,
                 n_layer: Optional[int] = 24,
                 n_head: Optional[int] = 16,
                 do_layer_norm_before: Optional[bool] = True,
                 word_embed_proj_dim: Optional[int] = None,
                 dropout: Optional[float] = 0.1,
                 dropatt: Optional[float] = 0.0,
                 activation_dropout: Optional[float] = 0.0,
                 layerdrop: Optional[float] = 0.0,
                 weight_init_std: Optional[float] = 0.02,
                 **kwargs) -> None:
        """Initializes the class by overriding default arguments.

        Args:
            n_token: Size of the vocabulary (number of tokens).
            tgt_len: Maximum length of sequences (positional embeddings).
            d_model: Dimensionality of the model (also for embedding layer).
            d_inner: Dimensionality of inner feed-forward layers.
            n_layer: Number of layers.
            n_head: Number of attention heads.
            do_layer_norm_before: Whether to perform normalization before attention block.
            word_embed_proj_dim: Dimensionality of word-embeddings projection.
            dropout: Feed-forward dropout probability.
            dropatt: Attention dropout probability.
            activation_dropout: Feed-forward activation dropout probability.
            layerdrop: Layer dropout probability.
            weight_init_std: Standard deviation to initialize the weights.

        """

        if not word_embed_proj_dim:
            word_embed_proj_dim = d_model 

        self.n_token = n_token
        self.tgt_len = tgt_len
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_layer = n_layer
        self.n_head = n_head
        self.do_layer_norm_before = do_layer_norm_before       
        self.word_embed_proj_dim = word_embed_proj_dim
        self.dropout = dropout
        self.dropatt = dropatt
        self.activation_dropout = activation_dropout
        self.layerdrop = layerdrop
        self.weight_init_std = weight_init_std

        additional_config = CONFIG_MAPPING['opt']().to_dict()
        for key, value in additional_config.items():
            if key not in self.__dict__.keys():
                setattr(self, key, value)

        super().__init__(**kwargs)


class HfOPTSearchConfig(SearchConfig):
    """Huggingface's Open Pre-Trained Transformer search configuration.

    """

    def __init__(self) -> None:
        """Initializes the class by setting default parameters that are used during search.
        
        """
        
        # Default HfOPT search options: n_layer, d_model, d_inner and n_head
        n_layer = SearchConfigParameter(per_layer=False, value=list(range(3, 30, 1)))
        d_model = SearchConfigParameter(per_layer=False, value=list(range(512, 1536, 64)))
        d_inner = SearchConfigParameter(per_layer=False, value=list(range(512, 6144, 64)))
        n_head = SearchConfigParameter(per_layer=False, value=[2, 4, 8, 16])

        super().__init__(n_layer=n_layer,
                         d_model=d_model,
                         d_inner=d_inner,
                         n_head=n_head)
