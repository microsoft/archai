# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Converts from parameters to genes and vice-versa.
"""

from collections import OrderedDict
from typing import Any, Dict, List


class Converter:
    """Enables conversion from genes to configuration dictionaries and vice-versa.

    """

    def __init__(self, **kwargs) -> None:
        """Overrides initialization method.

        """

        self.config = OrderedDict(kwargs)

        try:
            self.max_n_layer = max(self.config['n_layer']['value'])
        except:
            self.max_n_layer = 1

    def gene_to_config(self, gene: List[Any]) -> Dict[str, Any]:
        """Converts a gene into a configuration dictionary.

        Args:
            gene: Encoded gene.

        Returns:
            (Dict[str, Any]): Configuration dictionary.

        """

        config, idx = {}, 0

        for k, d in self.config.items():
            if d['per_layer']:
                config[k] = gene[idx:idx+self.max_n_layer]
                idx += self.max_n_layer
            else:
                config[k] = gene[idx]
                idx += 1

        return config

    def gene_to_key(self, gene: List[Any]) -> str:
        """Converts a gene into a configuration string.

        Args:
            gene: Encoded gene.

        Returns:
            (str): Configuration string.

        """

        config = self.gene_to_config(gene)
        n_layer = config['n_layer']

        key = []

        for v in config.values():
            if isinstance(v, list):
                key += v[:n_layer]
            else:
                key += [v]

        return ','.join(str(k) for k in key)

    def get_allowed_genes(self) -> List[List[Any]]:
        """Gathers all allowed gene choices.

        Returns:
            (List[List[Any]]): List of possible gene choices.
            
        """

        allowed_genes = []

        for d in self.config.values():
            if d['per_layer']:
                for _ in range(self.max_n_layer):
                    allowed_genes.append(d['value'])
            else:
                allowed_genes.append(d['value'])

        return allowed_genes
