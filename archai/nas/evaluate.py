from typing import Optional

import torch

from archai.common.trainer import Trainer
from archai.common.config import Config
from archai.common.common import logger
from archai.datasets import data
from .model_desc import ModelDesc
from .cell_builder import CellBuilder
from . import nas_utils

def eval_arch(conf_eval:Config, cell_builder:Optional[CellBuilder]):
    logger.pushd('eval_arch')

    # region conf vars
    conf_loader       = conf_eval['loader']
    save_filename    = conf_eval['save_filename']
    conf_model_desc   = conf_eval['model_desc']
    conf_checkpoint = conf_eval['checkpoint']
    resume = conf_eval['resume']
    conf_train = conf_eval['trainer']
    final_desc_filename = conf_eval['final_desc_filename']
    full_desc_filename = conf_eval['full_desc_filename']
    # endregion

    # load model desc file to get template model
    template_model_desc = ModelDesc.load(final_desc_filename)

    device = torch.device(conf_eval['device'])

    if cell_builder:
        cell_builder.register_ops()

    # TODO: move aux_tower, droppath into config
    model = nas_utils.model_from_conf(full_desc_filename,
                                conf_model_desc, device,
                                aux_tower=True, affine=True, droppath=True,
                                template_model_desc=template_model_desc)

    # get data
    train_dl, _, test_dl = data.get_data(conf_loader)
    assert train_dl is not None and test_dl is not None

    checkpoint = nas_utils.create_checkpoint(conf_checkpoint, resume)
    trainer = Trainer(conf_train, model, device, checkpoint, aux_tower=True)
    train_metrics = trainer.fit(train_dl, test_dl)
    train_metrics.save('eval_train_metrics')

    # save model
    save_path = model.save(save_filename)
    logger.info({'model_save_path': save_path})
    logger.popd()






