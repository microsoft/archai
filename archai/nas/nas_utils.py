# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple, Optional

from torch import nn
from torch.utils.data.dataloader import DataLoader

import tensorwatch as tw

from archai.common.config import Config
from archai.nas.model import Model
from archai.common.common import logger
from archai.common.checkpoint import CheckPoint



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

