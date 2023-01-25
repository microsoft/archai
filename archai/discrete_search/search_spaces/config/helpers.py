# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from typing import Any, Dict, List, Optional

from archai.discrete_search.search_spaces.config.discrete_choice import DiscreteChoice


def repeat_config(
    config_dict: Dict[str, Any], repeat_times: List[int], share_arch: Optional[bool] = False
) -> Dict[str, Any]:
    """Repeats a configuration multiple times.

    Args:
        config_dict: Configuration dictionary.
        repeat_times: Number of times to repeat the configuration.
        share_arch: Whether to share the architecture between the repeated configurations.

    Returns:
        Repeated configuration dictionary.

    """

    return {
        "_config_type": "config_list",
        "_share_arch": share_arch,
        "_repeat_times": DiscreteChoice(repeat_times),
        "_configs": {str(i): (config_dict if share_arch else deepcopy(config_dict)) for i in range(max(repeat_times))},
    }
