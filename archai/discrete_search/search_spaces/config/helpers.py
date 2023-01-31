# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

from archai.discrete_search.search_spaces.config.discrete_choice import DiscreteChoice


def repeat_config(
    config_dict: Dict[str, Any], repeat_times: Union[int, List[int]], share_arch: Optional[bool] = False
) -> Dict[str, Any]:
    """Repeats an architecture config a variable number of times.

    Args:
        config_dict (Dict[str, Any]): Config dictionary to repeat.
        
        repeat_times (Union[int, List[int]]): If an integer, the number of times to repeat the config 
            will be treated as constant. If a list of integers, the number of times to repeat the config will be
            also considered an architecture parameter and will be sampled from the list.
        
        share_arch (bool, optional): Whether to share the architecture parameters across the
            repeated configs. Defaults to False.

    Returns:
        Dict[str, Any]: Config dictionary with the repeated config.
    """    

    return {
        "_config_type": "config_list",
        "_share_arch": share_arch,
        "_repeat_times": DiscreteChoice(repeat_times),
        "_configs": {str(i): (config_dict if share_arch else deepcopy(config_dict)) for i in range(max(repeat_times))},
    }
