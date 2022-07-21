# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Huggingface's CodeGen configurations.
"""

from typing import Optional

from archai.nlp.models.config_base import Config, SearchConfigParameter, SearchConfig
from archai.nlp.models.hf_codegen.hf_codegen_utils.codegen_configuration import CodeGenConfig


class HfCodeGenConfig(Config):
    """Huggingface's CodeGen default configuration.

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
    attribute_map.update(CodeGenConfig.attribute_map)

    def __init__(self,
                 n_token: Optional[int] = 51200,
                 tgt_len: Optional[int] = 2048,
                 d_model: Optional[int] = 1024,
                 d_inner: Optional[int] = 4096,
                 n_layer: Optional[int] = 20,
                 n_head: Optional[int] = 16,
                 rotary_dim: Optional[int] = 32,
                 dropout: Optional[float] = 0.1,
                 dropatt: Optional[float] = 0.0,
                 embd_pdrop: Optional[float] = 0.1,
                 layer_norm_epsilon: Optional[float] = 0.00001,
                 scale_attn_weights: Optional[bool] = True,
                 weight_init_std: Optional[float] = 0.02,
                 tie_weight: Optional[bool] = False,
                 **kwargs) -> None:
        """Initializes the class by overriding default arguments.

        Args:
            n_token: Size of the vocabulary (number of tokens).
            tgt_len: Maximum length of sequences (positional embeddings).
            d_model: Dimensionality of the model (also for embedding layer).
            d_inner: Dimensionality of inner feed-forward layers.
            n_layer: Number of layers.
            n_head: Number of attention heads.
            rotary_dim: Dimensionality of rotary position embeddings.
            dropout: Feed-forward dropout probability.
            dropatt: Attention dropout probability.
            embd_pdrop: Embeddings dropout probabbility.
            layer_norm_epsilon: Epsilon used in layer normalization.
            scale_attn_weights: Whether to scale attention weights.
            weight_init_std: Standard deviation to initialize the weights.
            tie_weight: Whether embedding and softmax weights should be tied.

        """

        self.n_token = n_token
        self.tgt_len = tgt_len
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_layer = n_layer
        self.n_head = n_head
        self.rotary_dim = rotary_dim
        self.dropout = dropout
        self.dropatt = dropatt
        self.embd_pdrop = embd_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.scale_attn_weights = scale_attn_weights
        self.weight_init_std = weight_init_std
        self.tie_weight = tie_weight
        kwargs["tie_word_embeddings"] = tie_weight

        additional_config = CodeGenConfig().to_dict()
        for key, value in additional_config.items():
            if key not in self.__dict__.keys():
                setattr(self, key, value)

        super().__init__(**kwargs)


class HfCodeGenSearchConfig(SearchConfig):
    """Huggingface's CodeGen search configuration.

    """

    def __init__(self) -> None:
        """Initializes the class by setting default parameters that are used during search.
        
        """
        
        # Default HfCodeGen search options: n_layer, d_model, d_inner and n_head
        n_layer = SearchConfigParameter(per_layer=False, value=[3, 4, 5, 6, 7, 8, 9, 10])
        d_model = SearchConfigParameter(per_layer=False, value=list(range(128, 1024, 64)))
        d_inner = SearchConfigParameter(per_layer=False, value=list(range(128, 4096, 64)))
        n_head = SearchConfigParameter(per_layer=False, value=[2, 4, 8])

        super().__init__(n_layer=n_layer,
                         d_model=d_model,
                         d_inner=d_inner,
                         n_head=n_head)
