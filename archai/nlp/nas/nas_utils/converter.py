# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Converts from parameters to genes and vice-versa.
"""

from collections import OrderedDict
from typing import Any, Dict, List

# Defines keys that can be different for each layer
PER_LAYER_KEYS = ['d_inner', 'n_head']


class Converter:
    """Enables conversion from genes to configuration dictionaries and vice-versa.

    """

    def __init__(self, **kwargs) -> None:
        """Overrides initialization method.

        """

        self.config = OrderedDict(kwargs)

        try:
            self.max_n_layer = max(self.config['n_layer'])
        except:
            self.max_n_layer = 1

    def config_to_gene(self, config: Dict[str, Any]) -> List[Any]:
        """Converts a configuration dictionary into a gene.

        Args:
            config: Configuration dictionary.

        Returns:
            (List[Any]): Encoded gene ready for the search.

        """

        gene = []
        n_layer = config['n_layer']

        for k in self.config.keys():
            if k in PER_LAYER_KEYS:
                for i in range(max(self.max_n_layer, n_layer)):
                  if isinstance(config[k], list):
                    if i < n_layer:
                        gene.append(config[k][i])
                    else:
                        gene.append(config[k][0])
            else:
                gene.append(config[k])
                    
        return gene

    def gene_to_config(self, gene: List[Any]) -> Dict[str, Any]:
        """Converts a gene into a configuration dictionary.

        Args:
            gene: Encoded gene.

        Returns:
            (Dict[str, Any]): Configuration dictionary.

        """

        config = {}
        idx = 0

        for k in self.config.keys():
            if k in PER_LAYER_KEYS:
                config[k] = gene[idx:idx+self.max_n_layer]
                idx += self.max_n_layer
            else:
                config[k] = gene[idx]
                idx += 1

        return config

    def gene_to_str(self, gene: List[Any]) -> str:
        """Converts a gene into a configuration string.

        Args:
            gene: Encoded gene.

        Returns:
            (str): Configuration string.

        """

        return ','.join(str(g) for g in gene)

    def get_allowed_genes(self) -> List[List[Any]]:
        """Gathers all allowed gene choices.

        Returns:
            (List[List[Any]]): List of possible gene choices.
            
        """

        allowed_genes = []

        for k, v in self.config.items():
            if k in PER_LAYER_KEYS:
                for _ in range(self.max_n_layer):
                    allowed_genes.append(v)
            else:
                allowed_genes.append(v)

        return allowed_genes