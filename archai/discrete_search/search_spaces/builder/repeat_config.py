from typing import Dict, Any, List
from copy import deepcopy
from random import Random

from archai.discrete_search.search_spaces.builder.discrete_choice import DiscreteChoice

class RepeatConfig():
    def __init__(self, config_dict: Dict[str, Any], repeat_times: List[int], share_arch: bool = False):
        self.repeat_times = repeat_times
        self.share_arch = share_arch

        # If arch params should be shared between blocks, re-uses the same
        # reference to `config_dict` instead of creating a new copy
        self.config_dict = {
            '_config_type': 'config_list',
            '_share_arch': share_arch,
            '_repeat_times': DiscreteChoice(repeat_times),
            '_configs': {
                i: (config_dict if self.share_arch else deepcopy(config_dict))
                for i in range(max(repeat_times))
            },
        }

    def to_config_dict(self) -> Dict:
        return self.config_dict
