# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union, Optional, Tuple, List, Dict, Callable
from overrides import EnforceOverrides

from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.objective import Objective, AsyncObjective
from archai.discrete_search.api.dataset import DatasetProvider

import numpy as np
from tqdm import tqdm


class SearchObjectives(EnforceOverrides):
    def __init__(self, cache_objective_evaluation: bool = True, progress_bar: bool = True) -> None:
        self.cheap_objs = {}
        self.exp_objs = {}
        self.proxy_objs = {}

        self.progress_bar = progress_bar
        self.cache_objective_evaluation = cache_objective_evaluation
        
        # Cache key: (obj_name, is_proxy, archid, dataset obj, budget)
        self.cache: Dict[Tuple[str, bool, str, DatasetProvider, Optional[float]], Optional[float]] = {}

    @property
    def objs(self):
        return {
            k: v 
            for d in [self.cheap_objs, self.exp_objs, self.proxy_objs]
            for k, v in d.items() 
        }

    def add_cheap_objective(self, objective_name: str, objective: Union[Objective, AsyncObjective],
                            higher_is_better: bool,
                            constraint: Optional[Tuple[float, float]] = None) -> None:
        assert isinstance(objective, (AsyncObjective, Objective))
        assert objective_name not in self.objs, f'There is an objective named {objective_name} already.'

        self.cheap_objs[objective_name] = {
            'objective': objective,
            'higher_is_better': higher_is_better,
            'constraint': constraint or [-float('-inf'), float('+inf')],
            'proxy': False
        }
    
    def add_expensive_objective(self, objective_name: str,
                                objective: Union[Objective, AsyncObjective],
                                higher_is_better: bool,
                                constraint: Optional[Tuple[float, float]] = None,
                                proxy_constraint: Optional[Tuple[Union[Objective, AsyncObjective], float, float]] = None) -> None:
        assert isinstance(objective, (AsyncObjective, Objective))
        assert objective_name not in self.objs

        self.exp_objs[objective_name] = {
            'objective': objective,
            'higher_is_better': higher_is_better,
            'constraint': constraint or [-float('-inf'), float('+inf')],
            'proxy': False
        })

        if proxy_constraint:
            proxy_objective, *p_constraint = proxy_constraint

            self.proxy_objs[objective_name] = {
                'objective': proxy_objective,
                'higher_is_better': higher_is_better,
                'constraint': p_constraint,
                'proxy': True
            }

