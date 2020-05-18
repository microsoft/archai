from typing import Iterable, Tuple, Optional, Any, List
from collections import OrderedDict
import numpy as np
import yaml
import os

import torch
from torch import nn, Tensor

from overrides import overrides


from .cell import Cell
from .operations import Op, DropPath_
from .model_desc import ModelDesc, AuxTowerDesc, CellDesc
from ..common.common import logger, expdir_filepath
from ..common import utils, ml_utils

class Model(nn.Module):
    def __init__(self, model_desc:ModelDesc, droppath:bool, affine:bool):
        super().__init__()

        self.desc = model_desc
        self._stem0_op = Op.create(model_desc.stem0_op, affine=affine)
        self._stem1_op = Op.create(model_desc.stem1_op, affine=affine)

        self._cells = nn.ModuleList()
        self._aux_towers = nn.ModuleList()

        for i, (cell_desc, aux_tower_desc) in \
                enumerate(zip(model_desc.cell_descs(), model_desc.aux_tower_descs)):
            self._build_cell(cell_desc, aux_tower_desc, droppath, affine)

        # adaptive pooling output size to 1x1
        self._pool_op = Op.create(model_desc.pool_op, affine=affine)
        # since ch_p records last cell's output channels
        # it indicates the input channel number
        self._logits_op = Op.create(model_desc.logits_op, affine=affine)

        logger.info({'model_summary': self.summary()})

    def _build_cell(self, cell_desc:CellDesc,
                    aux_tower_desc:Optional[AuxTowerDesc],
                    droppath:bool, affine:bool)->None:
        alphas_cell = None if cell_desc.alphas_from==cell_desc.id  \
                            else self._cells[cell_desc.alphas_from]
        cell = Cell(cell_desc,
                    affine=affine, droppath=droppath,
                    alphas_cell=alphas_cell)
        self._cells.append(cell)
        self._aux_towers.append(AuxTower(aux_tower_desc, pool_stride=3) \
                                if aux_tower_desc else None)

    def summary(self)->dict:
        return {
            'cell_count': len(self._cells),
            #'cell_params': [ml_utils.param_size(c) for c in self._cells]
            'params': ml_utils.param_size(self),
            'alphas_p': len(list(a for a in self.alphas())),
            'alphas': np.sum(a.numel() for a in self.alphas()),
            'ops': np.sum(len(n.edges) for c in self.desc.cell_descs() for n in c.nodes()),
        }

    def alphas(self)->Iterable[nn.Parameter]:
        for cell in self._cells:
            if not cell.shared_alphas:
                for alpha in cell.alphas():
                    yield alpha

    def weights(self)->Iterable[nn.Parameter]:
        for cell in self._cells:
            for w in cell.weights():
                yield w

    def ops(self)->Iterable[Op]:
        for cell in self._cells:
            for op in cell.ops():
                yield op

    @overrides
    def forward(self, x)->Tuple[Tensor, Optional[Tensor]]:
        # TODO: original code has slighly different way of applying stems
        s0 = self._stem0_op(x)
        s1 = self._stem1_op(x)

        logits_aux = None
        for cell, aux_tower in zip(self._cells, self._aux_towers):
            #print(s0.shape, s1.shape, end='')
            s0, s1 = s1, cell.forward(s0, s1)
            #print('\t->\t', s0.shape, s1.shape)

            # TODO: this mimics darts but won't work for multiple aux towers
            if aux_tower is not None and self.training:
                logits_aux = aux_tower(s1)

        # s1 is now the last cell's output
        out = self._pool_op(s1)
        logits = self._logits_op(out) # flatten

        return logits, logits_aux

    def device_type(self)->str:
        return next(self.parameters()).device.type

    def finalize(self, to_cpu=True, restore_device=True)->ModelDesc:
        # move model to CPU before finalize because each op will serialize
        # its parameters and we don't want copy of these parameters lying on GPU
        original = self.device_type()
        if to_cpu:
            self.cpu()

        # finalize will create copy of state and this can overflow GPU RAM
        assert self.device_type() == 'cpu'

        cell_descs = [cell.finalize() for cell in self._cells]

        if restore_device:
            self.to(original, non_blocking=True)

        return ModelDesc(stem0_op=self._stem0_op.finalize()[0],
                         stem1_op=self._stem1_op.finalize()[0],
                         pool_op=self._pool_op.finalize()[0],
                         ds_ch=self.desc.ds_ch,
                         n_classes=self.desc.n_classes,
                         cell_descs=cell_descs,
                         aux_tower_descs=self.desc.aux_tower_descs,
                         logits_op=self._logits_op.finalize()[0],
                         params=self.desc.params)

    def drop_path_prob(self, p:float):
        """ Set drop path probability
        This will be called externally so any DropPath_ modules get
        new probability. Typically, every epoch we will reduce this probability.
        """
        for module in self.modules():
            if isinstance(module, DropPath_):
                module.p = p

    def save(self, filename:str, subdir:List[str]=[])->Optional[str]:
        save_path = expdir_filepath(filename, subdir)
        if save_path:
            ml_utils.save_model(self, save_path)
        return save_path


class AuxTower(nn.Module):
    def __init__(self, aux_tower_desc:AuxTowerDesc, pool_stride:int):
        """assuming input size 14x14"""
        # TODO: assert input size
        super().__init__()

        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=pool_stride, padding=0, count_include_pad=False),
            nn.Conv2d(aux_tower_desc.ch_in, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # TODO: This batchnorm was omitted in orginal implementation due to a typo.
            # nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # TODO: original code does't have adaptive pooling but things
            # don't work for variable cells otherwise
            nn.AdaptiveAvgPool2d(1)
        )
        self._logits_op = nn.Linear(768, aux_tower_desc.n_classes)

    def forward(self, x:torch.Tensor):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self._logits_op(x)
        return x