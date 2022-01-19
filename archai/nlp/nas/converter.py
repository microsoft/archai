# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Converts from parameters to genes and vice-versa.
"""

from typing import Any, Dict, List, Optional

import numpy as np

PER_LAYER_IDX = 2


class Converter:
    """Enables conversion from genes to configuration dictionaries and vice-versa.

    """

    def __init__(self,
                 n_layer_choice: int,
                 d_model_choice: int,
                 d_inner_choice: int,
                 n_head_choice: int) -> None:
        """Overrides initialization method.

        Args:
            n_layer_choice: Possible number of layers.
            d_model_choice: Possible model's dimension.
            d_inner_choice: Possible inner dimension.
            n_head_choice: Possible number of heads.

        """

        self.n_layer_choice = n_layer_choice
        self.d_model_choice = d_model_choice
        self.d_inner_choice = d_inner_choice
        self.n_head_choice = n_head_choice

        self.max_n_layer = self.n_layer_choice[-1]

    def config_to_gene(self, config: Dict[str, Any]) -> List[Any]:
        """Converts a configuration dictionary into a gene.

        Args:
            config: Configuration dictionary.

        Returns:
            (List[Any]): Encoded gene ready for the search.

        """

        gene = []

        sample_n_layer = config['n_layer']

        gene.append(config['d_model'])
        gene.append(sample_n_layer)

        for i in range(max(self.max_n_layer, sample_n_layer)):
            if isinstance(config['d_inner'], list):
                if i < sample_n_layer:
                    gene.append(config['d_inner'][i])
                else:
                    gene.append(config['d_inner'][0])
            else:
                gene.append(config['d_inner'])

        for i in range(max(self.max_n_layer, sample_n_layer)):
            if isinstance(config['n_head'], list):
                if i < sample_n_layer:
                    gene.append(config['n_head'][i])
                else:
                    gene.append(config['n_head'][0])
            else:
                gene.append(config['n_head'])

        return gene

    def gene_to_config(self, gene: List[Any]) -> Dict[str, Any]:
        """Converts a gene into a configuration dictionary.

        Args:
            gene: Encoded gene.

        Returns:
            (Dict[str, Any]): Configuration dictionary.

        """

        config = {'d_model': None,
                  'n_layer': None,
                  'd_inner': None,
                  'n_head': None}

        current_index = 0

        config['d_model'] = gene[current_index]
        current_index += 1

        config['n_layer'] = gene[current_index]
        current_index += 1

        config['d_inner'] = gene[current_index: current_index + config['n_layer']]
        current_index += max(self.max_n_layer, config['n_layer'])

        config['n_head'] = gene[current_index: current_index + config['n_layer']]
        current_index += max(self.max_n_layer, config['n_layer'])

        return config

    def gene_to_str(self, gene: List[Any]) -> str:
        """Converts a gene into a configuration string.

        Args:
            gene: Encoded gene.

        Returns:
            (str): Configuration string.

        """

        key_list = []

        current_index = 0

        key_list += [gene[current_index]]  # d_model
        current_index += 1

        key_list += [gene[current_index]]  # n_layer
        current_index += 1

        key_list += gene[current_index: current_index + gene[1]]  # d_inner
        current_index += self.max_n_layer

        key_list += gene[current_index: current_index + gene[1]]  # n_head
        current_index += self.max_n_layer

        return ','.join(str(k) for k in key_list)

    def get_allowed_genes(self, d_inner_min: Optional[int] = None) -> List[List[Any]]:
        """Gathers all allowed gene choices.

        Args:
            d_inner_min: Minimum value for the inner dimension.

        Returns:
            (List[List[Any]]): List of possible gene choices.
            
        """

        allowed_genes = []

        allowed_genes.append(self.d_model_choice)
        allowed_genes.append(self.n_layer_choice)

        for _ in range(self.max_n_layer):
            if d_inner_min is not None:
                allowed_genes.append(list(range(d_inner_min, self.d_inner_choice[-1], 50)))
            else:
                allowed_genes.append(self.d_inner_choice)

        for _ in range(self.max_n_layer):
            allowed_genes.append(self.n_head_choice)

        return allowed_genes


def params_to_config(params, max_n_layer, x):
        """
        """

        #
        x = np.squeeze(x, -1)

        #
        param_idx = 0
        config = {}

        #
        for k, v in params.items():
            if not v[PER_LAYER_IDX]:
                config[k] = np.round(x[param_idx]).astype(int)
                param_idx += 1
            else:
                config[k] = np.round(x[param_idx: param_idx + max_n_layer]).astype(int).tolist()
                param_idx += max_n_layer

        #
        for k, v in config.items():
            if isinstance(v, list):
                config[k] = config[k][:config['n_layer']]

        return config
