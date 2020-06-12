# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple, Optional

from torch import nn
from torch.utils.data.dataloader import DataLoader

from .model_desc import ModelDesc
from .macro_builder import MacroBuilder
from .cell_builder import CellBuilder
from ..common.config import Config
from .model import Model
from ..common.common import logger
from ..common.checkpoint import CheckPoint

def build_cell(model_desc, cell_builder:Optional[CellBuilder], search_iter:int)->None:
    if cell_builder:
        cell_builder.register_ops()
        cell_builder.build(model_desc, search_iter)

def create_macro_desc(conf_model_desc: Config,
                      template_model_desc:Optional[ModelDesc])->ModelDesc:
    builder = MacroBuilder(conf_model_desc, template=template_model_desc)
    model_desc = builder.build()
    return model_desc

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

def model_from_conf(full_desc_filename:str, conf_model_desc: Config,
        affine:bool, droppath:bool, template_model_desc:ModelDesc)->Model:
    """Creates model given desc config and template"""
    # create model
    model_desc = create_macro_desc(conf_model_desc, template_model_desc)
    # save model that we would eval for reference
    model_desc.save(full_desc_filename)

    return model_from_desc(model_desc, droppath=droppath, affine=affine)

def model_from_desc(model_desc, droppath:bool, affine:bool)->Model:
    model = Model(model_desc, droppath=droppath, affine=affine)
    return model


