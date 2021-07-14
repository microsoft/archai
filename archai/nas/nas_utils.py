# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple, Optional
from collections import namedtuple

from torch import nn
from torch.utils.data.dataloader import DataLoader

import tensorwatch as tw

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
            


