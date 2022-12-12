from typing import Dict, List, Union

import numpy as np
from tqdm import tqdm

from archai.discrete_search import ArchaiModel, AsyncObjective, Objective


def get_pareto_frontier(models: List[ArchaiModel], 
                        evaluation_results: Dict[str, np.ndarray],
                        objectives: Dict[str, Union[Objective, AsyncObjective]]) -> Dict:
    assert len(objectives) == len(evaluation_results)
    assert all(len(r) == len(models) for r in evaluation_results.values())

    # Inverts maximization objectives 
    inverted_results = {
        obj_name: (-obj_results if objectives[obj_name].higher_is_better else obj_results)
        for obj_name, obj_results in evaluation_results.items()
    }

    # Converts results to an array of shape (len(models), len(objectives))
    results_array = np.vstack(list(inverted_results.values())).T

    pareto_points = np.array(
        _find_pareto_frontier_points(results_array)
    )

    return {
        'models': [models[idx] for idx in pareto_points],
        'evaluation_results': {
            obj_name: obj_results[pareto_points]
            for obj_name, obj_results in evaluation_results.items()
        },
        'indices': pareto_points
    }


def get_non_dominated_sorting(models: List[ArchaiModel],
                              evaluation_results: Dict[str, np.ndarray],
                              objectives: Dict[str, Union[Objective, AsyncObjective]]) -> List[Dict]:
    assert len(objectives) == len(evaluation_results)
    assert all(len(r) == len(models) for r in evaluation_results.values())

    # Inverts maximization objectives 
    inverted_results = {
        obj_name: (-obj_results if objectives[obj_name].higher_is_better else obj_results)
        for obj_name, obj_results in evaluation_results.items()
    }

    # Converts results to an array of shape (len(models), len(objectives))
    results_array = np.vstack(list(inverted_results.values())).T

    frontiers = np.array(
        _find_non_dominated_sorting(results_array)
    )

    return [{
        'models': [models[idx] for idx in frontier],
        'evaluation_results': {
            obj_name: obj_results[frontier]
            for obj_name, obj_results in evaluation_results.items()
        },
        'indices': frontier
    } for frontier in frontiers]


def _find_pareto_frontier_points(all_points: np.ndarray) -> List[int]:
    """Takes in a list of n-dimensional points, one per row, returns the list of row indices
        which are Pareto-frontier points.
        
    Assumes that lower values on every dimension are better.

    Args:
        all_points: N-dimensional points.

    Returns:
        List of Pareto-frontier indexes.
    """

    # For each point see if there exists  any other point which dominates it on all dimensions
    # If that is true, then it is not a pareto point and vice-versa

    # Inputs should alwyas be a two-dimensional array
    assert len(all_points.shape) == 2

    pareto_inds = []

    dim = all_points.shape[1]

    # Gets the indices of unique points
    _, unique_indices = np.unique(all_points, axis=0, return_index=True)

    for i in unique_indices:
        this_point = all_points[i,:]
        is_pareto = True

        for j in unique_indices:
            if j == i:
                continue

            other_point = all_points[j,:]
            diff = this_point - other_point

            if sum(diff >= 0) == dim:
                # Other point is smaller/larger on all dimensions
                # so we have found at least one dominating point
                is_pareto = False
                break
                
        if is_pareto:
            pareto_inds.append(i)

    return pareto_inds


def _find_non_dominated_sorting(all_points: np.ndarray) -> List[List[int]]:
    """Finds non-dominated sorting frontiers from a matrix (#points, #objectives).

    Args:
        all_points (np.ndarray): N-dimensional points.

    Returns:
        List[List[int]]: List of frontier indices
    
    References:
        Implementation adapted from:
            https://github.com/anyoptimization/pymoo/blob/main/pymoo/util/nds/efficient_non_dominated_sort.py
    
        Algorithm:
            X. Zhang, Y. Tian, R. Cheng, and Y. Jin,
            An efficient approach to nondominated sorting for evolutionary multiobjective optimization,
            IEEE Transactions on Evolutionary Computation, 2015, 19(2): 201-213.
    """

    lex_sorting = np.lexsort(all_points.T[::-1])
    all_points = all_points.copy()[lex_sorting]

    fronts = []

    for idx in range(all_points.shape[0]):
        front_rank = _find_front_rank(all_points, idx, fronts)        
        
        if front_rank >= len(fronts):
            fronts.append([])

        fronts[front_rank].append(idx)

    ret = []
    for front in fronts:
        ret.append(lex_sorting[front])

    return ret


def _find_front_rank(all_points: np.ndarray, idx: int, fronts: List[List[int]]) -> int:
    """Finds the front rank for all_points[idx] given `fronts`.

    Args:
        all_points (np.ndarray): N-dimensional points.
        idx (int): Point index.
        fronts (List[List[int]]): Current NDS fronts

    Returns:
        int: Front rank for `all_points[idx]`
    
    Reference:
        Adapted from https://github.com/anyoptimization/pymoo/blob/main/pymoo/util/nds/efficient_non_dominated_sort.py
    """
    
    def dominates(x, y):
        for i in range(len(x)):
            if y[i] < x[i]:
                return False

        return True

    num_found_fronts = len(fronts)
    rank = 0
    current = all_points[idx]

    while True:
        if num_found_fronts == 0:
            return 0

        fk_indices = fronts[rank]
        solutions = all_points[fk_indices[::-1]]
        non_dominated = True

        for s in solutions:
            if dominates(s, current):
                non_dominated = False
                break

        if non_dominated:
            return rank
        else:
            rank += 1
            if rank >= num_found_fronts:
                return num_found_fronts
