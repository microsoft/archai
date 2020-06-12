# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Tuple, List

from overrides import EnforceOverrides

from ..common.config import Config
from .model_desc import ModelDesc, OpDesc, CellType, NodeDesc, EdgeDesc, \
                        CellDesc, AuxTowerDesc, ConvMacroParams
from ..common.common import logger
from .operations import ModelStemBase, Op

class MacroBuilder(EnforceOverrides):
    def __init__(self, conf_model_desc: Config,
                 template:Optional[ModelDesc]=None)->None:
        # region conf vars
        conf_data = conf_model_desc['dataset']
        self.ds_name = conf_data['name']
        self.ds_ch = conf_data['channels']
        self.n_classes = conf_data['n_classes']
        self.init_node_ch = conf_model_desc['init_node_ch']
        self.aux_tower_stride = conf_model_desc['aux_tower_stride']
        self.stem_multiplier = conf_model_desc['stem_multiplier']
        self.aux_weight = conf_model_desc['aux_weight']
        self.max_final_edges = conf_model_desc['max_final_edges']
        self.cell_post_op = conf_model_desc['cell_post_op']
        self.model_stem0_op = conf_model_desc['model_stem0_op']
        self.model_stem1_op = conf_model_desc['model_stem1_op']
        self.model_post_op = conf_model_desc['model_post_op']

        self.n_cells = conf_model_desc['n_cells']
        self.n_nodes = conf_model_desc['n_nodes']
        self.n_reductions = conf_model_desc['n_reductions']
        self.model_desc_params = conf_model_desc['params']
        # endregion

        # this satisfies N R N R N pattern
        assert self.n_cells >= self.n_reductions * 2 + 1

        # for each reduction, we create one indice
        # for cifar and imagenet, reductions=2 creating cuts at n//3, n*2//3
        self._reduction_cell_indices = \
            [self.n_cells*(i+1) // (self.n_reductions+1) \
                for i in range(self.n_reductions)]

        self._set_templates(template)

    def _set_templates(self, template:Optional[ModelDesc])->None:
        self.template = template
        self.normal_template,  self.reduction_template = None, None
        if self.template is not None:
            # find first regular and reduction cells and set them as
            # the template that we will use. When we create new cells
            # we will fill them up with nodes from these templates
            for cell_desc in self.template.cell_descs():
                if self.normal_template is None and \
                        cell_desc.cell_type==CellType.Regular:
                  self.normal_template = cell_desc
                if self.reduction_template is None and \
                        cell_desc.cell_type==CellType.Reduction:
                    self.reduction_template = cell_desc

    def build(self)->ModelDesc:
        # create model stems
        stem0_op, stem1_op = self._create_model_stems()

        # both stems should have same channel outs (same true for cell outputs)
        assert stem0_op.params['conv'].ch_out == stem1_op.params['conv'].ch_out

        # get reduction factors in spatial dimensions
        stem0_reduction, stem1_reduction = MacroBuilder._stem_reductions(stem0_op, stem1_op)

        # create cell descriptions
        cell_descs, aux_tower_descs = self._get_cell_descs(
            stem0_reduction < stem1_reduction,
            stem0_op.params['conv'].ch_out,
            self.max_final_edges)

        # get last cell channels
        if len(cell_descs):
            last_ch_out = cell_descs[-1].cell_ch_out
        else:
            last_ch_out = stem1_op.params['conv'].ch_out

        pool_op = OpDesc(self.model_post_op,
                         params={'conv': ConvMacroParams(last_ch_out, last_ch_out)},
                         in_len=1, trainables=None)

        logits_op = OpDesc('linear',
                        params={'n_ch':last_ch_out,'n_classes': self.n_classes}, in_len=1,
                        trainables=None)
        return ModelDesc(stem0_op, stem1_op, pool_op, self.ds_ch,
                         self.n_classes, cell_descs, aux_tower_descs,
                         logits_op, self.model_desc_params.to_dict())

    def _get_cell_descs(self, reduction_p, stem_ch_out:int, max_final_edges:int)\
            ->Tuple[List[CellDesc], List[Optional[AuxTowerDesc]]]:

        cell_descs, aux_tower_descs = [], []

        # chennels of prev-prev whole cell, prev whole cell and current cell node
        pp_ch_out, p_ch_out, node_ch_out = stem_ch_out, stem_ch_out, self.init_node_ch

        # stores first cells of each time with whom arch params would be shared
        first_normal, first_reduction = -1, -1

        for ci in range(self.n_cells):
            # find cell type and output channels for this cell
            # also update if this is our first cell from which arch params will be shared
            reduction = self._is_reduction(ci)
            if reduction:
                node_ch_out, cell_type = node_ch_out*2, CellType.Reduction
                first_reduction = ci if first_reduction < 0 else first_reduction
                template_cell = first_reduction
            else:
                cell_type = CellType.Regular
                first_normal = ci if first_normal < 0 else first_normal
                template_cell = first_normal

            s0_op, s1_op = self._get_cell_stems(
                node_ch_out, p_ch_out, pp_ch_out, reduction_p)

            # cell template for this cell contains nodes we can use as template
            cell_template = self._cell_template(cell_type)

            # cell template is not available in search time in which case we
            # will take number of nodes from config
            node_count = len(cell_template.nodes()) if cell_template \
                                                    else self.n_nodes

            # create nodes with empty edges
            nodes:List[NodeDesc] =  [NodeDesc(edges=[])
                                     for _ in range(node_count)]

            cell_descs.append(CellDesc(
                cell_type=cell_type, id=ci,
                nodes=nodes,
                s0_op=s0_op, s1_op=s1_op,
                template_cell=template_cell,
                max_final_edges=max_final_edges,
                node_ch_out=node_ch_out,
                post_op=self.cell_post_op
            ))
            # add any nodes from the template to the just added cell
            self._copy_template_nodes(cell_template, cell_descs[ci])
            # add aux tower
            aux_tower_descs.append(self._get_aux_tower(cell_descs[ci], ci))

            # we concate all channels so next cell node gets channels from all nodes
            pp_ch_out, p_ch_out = p_ch_out, cell_descs[ci].cell_ch_out
            reduction_p = reduction

        return cell_descs, aux_tower_descs

    def _cell_template(self, cell_type:CellType)->Optional[CellDesc]:
        if self.template is None:
            return None

        # select cell template
        reduction = cell_type == CellType.Reduction
        return self.reduction_template if reduction else self.normal_template

    def _copy_template_nodes(self, cell_template:Optional[CellDesc],
                             cell_desc:CellDesc)->None:
        if cell_template is None:
            return

        assert cell_template.cell_type==cell_desc.cell_type and \
               len(cell_template.nodes()) == len(cell_desc.nodes())

        # copy each template node to cell
        for node, node_t in zip(cell_desc.nodes(), cell_template.nodes()):
            edges_copy = [e.clone(
                            # use new macro params
                            conv_params=cell_desc.conv_params.clone(),
                            # TODO: check for compatibility?
                            clear_trainables=True
                            ) for e in node_t.edges]
            node.edges=edges_copy

    def _is_reduction(self, cell_index:int)->bool:
        # For darts, n_cells=8 so we build [N N R N N R N N] structure
        # Notice that this will result in only 2 reduction cells no matter
        # total number of cells. Original resnet actually have 3 reduction cells.
        # Between two reduction cells we have regular cells.
        return cell_index in self._reduction_cell_indices

    def _get_cell_stems(self, node_ch_out: int, p_ch_out: int, pp_ch_out:int,
                   reduction_p: bool)->Tuple[OpDesc, OpDesc]:

        # Cell stemps will take prev channels and out sameput channels as nodes would.
        # If prev cell was reduction then we need to increase channels of prev-prev
        # by 2X. This is done by prepr_reduce stem.
        s0_op = OpDesc('prepr_reduce' if reduction_p else 'prepr_normal',
                    params={
                        'conv': ConvMacroParams(pp_ch_out, node_ch_out)
                    }, in_len=1, trainables=None)

        s1_op = OpDesc('prepr_normal',
                    params={
                        'conv': ConvMacroParams(p_ch_out, node_ch_out)
                    }, in_len=1, trainables=None)
        return s0_op, s1_op

    def _get_aux_tower(self, cell_desc:CellDesc, cell_index:int)->Optional[AuxTowerDesc]:
        # TODO: shouldn't we be adding aux tower at *every* 1/3rd?
        if self.aux_weight and cell_index == 2*self.n_cells//3:
            return AuxTowerDesc(cell_desc.cell_ch_out, self.n_classes, self.aux_tower_stride)
        return None

    def _create_model_stems(self)->Tuple[OpDesc, OpDesc]:
        # TODO: weired not always use two different stemps as in original code
        # TODO: why do we need stem_multiplier?
        # TODO: in original paper stems are always affine
        conv_params = ConvMacroParams(self.ds_ch,
                                      self.init_node_ch*self.stem_multiplier)
        stem0_op = OpDesc(name=self.model_stem0_op, params={'conv': conv_params},
                          in_len=1, trainables=None)
        stem1_op = OpDesc(name=self.model_stem1_op, params={'conv': conv_params},
                          in_len=1, trainables=None)
        return stem0_op, stem1_op

    @staticmethod
    def _stem_reductions(stem0_op:OpDesc, stem1_op:OpDesc)->Tuple[int, int]:
        op1, op2 = Op.create(stem0_op, affine=False), Op.create(stem1_op, affine=False)
        assert isinstance(op1, ModelStemBase) and isinstance(op2, ModelStemBase)
        return op1.reduction, op2.reduction