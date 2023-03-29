# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import itertools
from collections import OrderedDict
from copy import deepcopy
from functools import reduce
from random import Random
from typing import Any, Dict, List, Optional, Tuple

from archai.discrete_search.search_spaces.config.arch_config import (
    ArchConfig,
    build_arch_config,
)
from archai.discrete_search.search_spaces.config.discrete_choice import DiscreteChoice
from archai.discrete_search.search_spaces.config.utils import flatten_dict, replace_ptree_choices, order_dict_keys


class ArchParamTree:
    """Tree of architecture parameters."""

    def __init__(self, config_tree: Dict[str, Any]) -> None:
        """Initialize the class.

        Args:
            config_tree: Tree of architecture parameters.

        """

        self.config_tree = deepcopy(config_tree)
        self.params, self.constants = self._init_tree(config_tree)

    @property
    def num_archs(self) -> int:
        """Return the number of architectures in the search space."""

        param_dict = self.to_dict(flatten=True, deduplicate_params=True, remove_constants=True)
        num_options = [float(len(p.choices)) for p in param_dict.values()]

        return reduce(lambda a, b: a*b, num_options, 1)

    def _init_tree(self, config_tree: Dict[str, Any]) -> Tuple[OrderedDict, OrderedDict]:
        params, constants = OrderedDict(), OrderedDict()

        for param_name, param in config_tree.items():
            if isinstance(param, DiscreteChoice):
                params[param_name] = param
            elif isinstance(param, dict):
                params[param_name] = ArchParamTree(param)
            else:
                constants[param_name] = param

        return params, constants

    def _to_dict(
        self, prefix: str, flatten: bool, dedup_param_ids: Optional[set] = None, remove_constants: Optional[bool] = True
    ) -> OrderedDict:
        prefix = f"{prefix}." if prefix else prefix
        output_dict = OrderedDict()

        # if not `remove_constants`, initializes the output
        # dictionary with constants first
        if not remove_constants:
            output_dict = OrderedDict(
                [
                    (prefix + c_name if flatten else c_name, c_value)
                    for c_name, c_value in deepcopy(self.constants).items()
                ]
            )

        # Adds architecture parameters to the output dictionary
        for param_name, param in self.params.items():
            param_name = prefix + str(param_name) if flatten else str(param_name)

            if isinstance(param, ArchParamTree):
                param_dict = param._to_dict(param_name, flatten, dedup_param_ids, remove_constants)

                if flatten:
                    output_dict.update(param_dict)
                else:
                    output_dict[param_name] = param_dict

            elif isinstance(param, DiscreteChoice):
                if dedup_param_ids is None:
                    output_dict[param_name] = param
                elif id(param) not in dedup_param_ids:
                    output_dict[param_name] = param
                    dedup_param_ids.add(id(param))

        return output_dict

    def to_dict(
        self,
        flatten: Optional[bool] = False,
        deduplicate_params: Optional[bool] = False,
        remove_constants: Optional[bool] = False,
    ) -> OrderedDict:
        """Convert the `ArchParamTree` to an ordered dictionary.

        Args:
            flatten: If the output dictionary should be flattened.
            deduplicate_params: Removes duplicate architecture parameters.
            remove_constants: Removes attributes that are not architecture params from
                the output dictionary.

        Returns:
            Ordered dictionary of architecture parameters.

        """

        return self._to_dict("", flatten, set() if deduplicate_params else None, remove_constants)

    def sample_config(self, rng: Optional[Random] = None) -> ArchConfig:
        """Sample an architecture config from the search param tree.

        Args:
            rng: Random number generator used during sampling. If set to `None`,
                `random.Random()` is used.

        Returns:
            Sampled architecture config.

        """

        rng = rng or Random()
        choices_dict = replace_ptree_choices(self.to_dict(), lambda x: x.random_sample(rng))

        return build_arch_config(choices_dict)

    def get_param_name_list(self) -> List[str]:
        """Get list of parameter names in the search space.

        Returns:
            List of parameter names.

        """

        param_dict = self.to_dict(flatten=True, deduplicate_params=True, remove_constants=True)

        return list(param_dict.keys())

    def encode_config(self, config: ArchConfig, track_unused_params: Optional[bool] = True) -> List[float]:
        """Encode an `ArchConfig` object into a fixed-length vector of features.

        This method should be used after the model object is created.

        Args:
            config: Architecture configuration.
            track_unused_params: If `track_unused_params=True`, parameters not used during
                model creation (by calling `config.pick`) will be represented as `float("NaN")`.

        Returns:
            List of features.

        """

        deduped_features = self.to_dict(flatten=True, deduplicate_params=True, remove_constants=True)

        flat_config = flatten_dict(config._config_dict)
        flat_used_params = flatten_dict(config.get_used_params())

        # Reorder `flat_config` and `flat_used_params` to follow the order of `deduped_features`
        flat_config = order_dict_keys(deduped_features, flat_config)
        flat_used_params = order_dict_keys(deduped_features, flat_used_params)

        # Build feature array
        features = OrderedDict([
            (k, deduped_features[k].encode(v)) 
            for k, v in flat_config.items() 
            if k in deduped_features
        ])

        # Replaces unused params with NaNs if necessary
        if track_unused_params:
            for feature_name, enc_param in features.items():
                if not flat_used_params[feature_name]:
                    features[feature_name] = [float("NaN") for _ in enc_param]

        # Flattens the feature array
        return list(itertools.chain(*features.values()))
