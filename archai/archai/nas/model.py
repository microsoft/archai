# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Iterable, Tuple, Optional, Any, List
from collections import OrderedDict
import numpy as np
import yaml
import os

import torch
from torch import nn, Tensor
from overrides import overrides

from archai.nas.arch_params import ArchParams
from archai.nas.cell import Cell
from archai.nas.operations import Op, DropPath_
from archai.nas.model_desc import ModelDesc, AuxTowerDesc, CellDesc
from archai.common.common import logger
from archai.common import utils, ml_utils
from archai.nas.arch_module import ArchModule

class Model(ArchModule):
    def __init__(self, model_desc:ModelDesc, droppath:bool, affine:bool):
        super().__init__()

        # some of these fields are public as finalizer needs access to them
        self.desc = model_desc

        # TODO: support any number of stems
        assert len(model_desc.model_stems)==2, "Model compiler currently only supports 2 stems"
        stem0_op = Op.create(model_desc.model_stems[0], affine=affine)
        stem1_op = Op.create(model_desc.model_stems[1], affine=affine)
        self.model_stems = nn.ModuleList((stem0_op, stem1_op))

        self.cells = nn.ModuleList()
        self._aux_towers = nn.ModuleList()

        for i, (cell_desc, aux_tower_desc) in \
                enumerate(zip(model_desc.cell_descs(), model_desc.aux_tower_descs)):
            self._build_cell(cell_desc, aux_tower_desc, droppath, affine)

        # adaptive pooling output size to 1x1
        self.pool_op = Op.create(model_desc.pool_op, affine=affine)
        # since ch_p records last cell's output channels
        # it indicates the input channel number
        self.logits_op = Op.create(model_desc.logits_op, affine=affine)

        # for i,cell in enumerate(self.cells):
        #     print(i, ml_utils.param_size(cell))
        #logger.info({'model_summary': self.summary()})

    def _build_cell(self, cell_desc:CellDesc,
                    aux_tower_desc:Optional[AuxTowerDesc],
                    droppath:bool, affine:bool)->None:
        trainables_from = None if cell_desc.trainables_from==cell_desc.id  \
                            else self.cells[cell_desc.trainables_from]
        cell = Cell(cell_desc, affine=affine, droppath=droppath,
                    trainables_from=trainables_from)
        self.cells.append(cell)
        self._aux_towers.append(AuxTower(aux_tower_desc) \
                                if aux_tower_desc else None)

    def summary(self)->dict:
        all_arch_params = list(self.all_owned()
                               .param_by_kind(kind=None))
        return {
            'cell_count': len(self.cells),
            #'cell_params': [ml_utils.param_size(c) for c in self.cells]
            'params': ml_utils.param_size(self),
            'arch_params_len': len(all_arch_params),
            'arch_params_numel': np.sum(a.numel() for a in all_arch_params),
            'ops': np.sum(len(n.edges) for c in self.desc.cell_descs() for n in c.nodes()),
        }

    def ops(self)->Iterable[Op]:
        for cell in self.cells:
            for op in cell.ops():
                yield op

    @overrides
    def forward(self, x)->Tuple[Tensor, Optional[Tensor]]:
        #print(torch.cuda.memory_allocated()/1.0e6)
        s0 = self.model_stems[0](x)
        #print(torch.cuda.memory_allocated()/1.0e6)
        s1 = self.model_stems[1](x)
        #print(-1, s0.shape, s1.shape, torch.cuda.memory_allocated()/1.0e6)

        logits_aux = None
        for ci, (cell, aux_tower) in enumerate(zip(self.cells, self._aux_towers)):
            #print(s0.shape, s1.shape, end='')
            s0, s1 = s1, cell.forward(s0, s1)
            #print(ci, s0.shape, s1.shape, torch.cuda.memory_allocated()/1.0e6)

            # TODO: this mimics darts but won't work for multiple aux towers
            if aux_tower is not None and self.training:
                logits_aux = aux_tower(s1)
                #print(ci, 'aux', logits_aux.shape)

        # s1 is now the last cell's output
        out = self.pool_op(s1)
        logits = self.logits_op(out) # flatten

        #print(-1, 'out', out.shape)
        #print(-1, 'logits', logits.shape)

        return logits, logits_aux

    def device_type(self)->str:
        return next(self.parameters()).device.type

    def drop_path_prob(self, p:float):
        """ Set drop path probability
        This will be called externally so any DropPath_ modules get
        new probability. Typically, every epoch we will reduce this probability.
        """
        for module in self.modules():
            if isinstance(module, DropPath_):
                module.p = p


class AuxTower(nn.Module):
    def __init__(self, aux_tower_desc:AuxTowerDesc):
        """assuming input size 14x14"""
        # TODO: assert input size?
        super().__init__()

        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=aux_tower_desc.stride, padding=0, count_include_pad=False),
            nn.Conv2d(aux_tower_desc.ch_in, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # TODO: This batchnorm was omitted in orginal implementation due to a typo.
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.logits_op = nn.Linear(768, aux_tower_desc.n_classes)

    def forward(self, x:torch.Tensor):
        x = self.features(x)
        x = self.logits_op(x.view(x.size(0), -1))
        return x