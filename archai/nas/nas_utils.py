# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Optional
from collections import namedtuple

from torch import nn
from torch.utils.data.dataloader import DataLoader

import tensorwatch as tw
import numpy as np

from archai.common.config import Config
from archai.nas.model import Model
from archai.common.common import logger
from archai.common.checkpoint import CheckPoint
from archai.nas.model_desc import ModelDesc, CellType



def checkpoint_empty(checkpoint:Optional[CheckPoint])->bool:
    return checkpoint is None or checkpoint.is_empty()

def create_checkpoint(conf_checkpoint:Config, resume:bool)->Optional[CheckPoint]:
    """Creates checkpoint given its config. If resume is True then attempt is
    made to load existing checkpoint otherwise an empty checkpoint is created.
    """
    checkpoint = CheckPoint(conf_checkpoint, resume) \
                 if conf_checkpoint is not None else None

    logger.info({'checkpoint_empty': checkpoint_empty(checkpoint),
                 'conf_checkpoint_none': conf_checkpoint is None, 'resume': resume,
                 'checkpoint_path': None  if checkpoint is None else checkpoint.filepath})
    return checkpoint

def get_model_stats(model:Model,
                    input_tensor_shape=[1,3,32,32], clone_model=True)->tw.ModelStats:
    # model stats is doing some hooks so do it last
    model_stats = tw.ModelStats(model, input_tensor_shape,
                                clone_model=clone_model)
    return model_stats


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

def create_nb301_genotype_from_desc(model_desc: ModelDesc)->Genotype:
    ''' Creates a Genotype object that can be used to 
    query Nasbench301 for training time and accuracy. 
    WARNING: the input ModelDesc must be compatible with nb301!'''
    info = dict()
    normal_concat = [2, 3, 4, 5]
    reduce_concat = [2, 3, 4, 5]
    info['normal_concat'] = normal_concat
    info['reduce_concat'] = reduce_concat

    for cell_desc in model_desc._cell_descs:
        edges_info = []
        assert len(cell_desc._nodes) == 4    
        for node in cell_desc._nodes:
            for edge in node.edges:
                edge_info = (edge.op_desc.name, edge.input_ids[0])
                edges_info.append(edge_info)

        assert len(edges_info) == 8

        if cell_desc.cell_type is CellType.Regular:
            info['normal'] = edges_info
        elif cell_desc.cell_type is CellType.Reduction:
            info['reduce'] = edges_info
            
    genotype = Genotype(
        normal = info['normal'],
        normal_concat = info['normal_concat'],
        reduce = info['reduce'],
        reduce_concat = info['reduce_concat']
    )

    return genotype
            

def find_pareto_frontier_points(all_points: np.ndarray,
                                is_decreasing: Optional[bool] = True) -> List[int]:
    """Takes in a list of n-dimensional points, one per row, returns the list of row indices
        which are Pareto-frontier points.
        
    Assumes that lower values on every dimension are better by default.

    Args:
        all_points: N-dimensional points.
        is_decreasing: Whether Pareto-frontier decreases or not.

    Returns:
        List of Pareto-frontier indexes.
    
    """

    # For each point see if there exists  any other point which dominates it on all dimensions
    # If that is true, then it is not a pareto point and vice-versa

    # Inputs should always be a two-dimensional array
    assert len(all_points.shape) == 2

    pareto_inds = []

    dim = all_points.shape[1]

    for i in range(all_points.shape[0]):
        this_point = all_points[i,:]
        is_pareto = True

        for j in range(all_points.shape[0]):
            if j == i:
                continue

            other_point = all_points[j,:]

            if is_decreasing:
                diff = this_point - other_point
            else:
                diff = other_point - this_point

            if sum(diff > 0) == dim:
                # Other point is smaller/larger on all dimensions
                # so we have found at least one dominating point
                is_pareto = False
                break
                
        if is_pareto:
            pareto_inds.append(i)

    return pareto_inds


def compute_crowding_distance(all_points: np.ndarray)->np.ndarray:
    '''Takes in a 2D numpy array, one point per row. Typically these
    are pareto-frontier points or estimate of pareto-frontier points. 
    Assumes each column is an objective in a multi-dimensional optimization problem.
    Computes the crowding distance for each point as detailed in 
    "An effective use of crowding distance in multiobjective particle swarm optimization"
    by Raquel et al., 2005."
    This function assumes that all the objectives (columns) are either increasing or decreasing.
    For example if considering latency, memory and accuracy, convert accuracy to error since
    that will make all objectives lower is better.

    Args:
        all_points: nd.array 2D numpy array.

    Returns: 
        c_dists: nd.array of shape (all_points.shape[0], 1)
    '''

    # Inputs should always be a two-dimensional array
    assert len(all_points.shape) == 2

    assert all_points.shape[0] > 0
    assert all_points.shape[1] > 0

    n = all_points.shape[0]
    num_objs = all_points.shape[1]

    c_dists = np.zeros((all_points.shape[0], 1))

    for i in range(num_objs):
        # sort in ascending order using this objective value
        ids = np.argsort(all_points[:, i])

        # set the distance to be the
        # distance to nearest neighbors on each side
        for j in range(1, n-1):
            row_num = ids[j]
            higher_nbr = ids[j+1]
            lower_nbr = ids[j-1]
            dist_btw_nbrs = all_points[higher_nbr, i] - all_points[lower_nbr, i]
            c_dists[row_num] += dist_btw_nbrs 
            
        # set the maximum distance to the boundary points
        # so that they are always selected
        c_dists[ids[0]] = np.inf
        c_dists[ids[-1]] = np.inf 

    assert c_dists.shape[0] == all_points.shape[0]
    return c_dists
