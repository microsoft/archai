# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Tuple, List
import copy

from overrides import EnforceOverrides

from archai.common.config import Config
from archai.nas.model_desc import ModelDesc, OpDesc, CellType, NodeDesc, EdgeDesc, \
                        CellDesc, AuxTowerDesc, ConvMacroParams
from archai.common.common import logger
from archai.nas.operations import StemBase, Op

# list of outputs for each module, each module has multiple tensors as output, each tensor shape is list
TensorShapes=List[List[List[int]]]

class ModelDescBuilder(EnforceOverrides):
    def __init__(self, conf_model_desc: Config,
                 template:Optional[ModelDesc]=None)->None:
        # region conf vars
        self.conf_data = conf_model_desc['dataset']
        self.ds_name = self.conf_data['name']
        self.ds_ch = self.conf_data['channels']
        self.n_classes = self.conf_data['n_classes']
        self.aux_tower_stride = conf_model_desc['aux_tower_stride']
        self.aux_weight = conf_model_desc['aux_weight']
        self.max_final_edges = conf_model_desc['max_final_edges']
        self.cell_post_op = conf_model_desc['cell_post_op']
        self.conf_model_stems = conf_model_desc['model_stems']
        self.model_post_op = conf_model_desc['model_post_op']

        self.n_cells = conf_model_desc['n_cells']
        self.n_nodes = conf_model_desc['n_nodes']
        self.n_reductions = conf_model_desc['n_reductions']
        self.model_desc_params = conf_model_desc['params']
        # endregion

        # for each reduction, we create one indice
        # for cifar and imagenet, reductions=2 creating cuts at n//3, n*2//3
        self._reduction_indices = self._get_reduction_indices()

        # if template model desc is specified thehn setup regular and reduction cell templates
        self._set_node_templates(template)

    def _get_reduction_indices(self)->set:
        # this satisfies N R N R N pattern
        assert self.n_cells >= self.n_reductions * 2 + 1

        # for each reduction, we create one indice
        # for cifar and imagenet, reductions=2 creating cuts at n//3, n*2//3
        return set(self.n_cells*(i+1) // (self.n_reductions+1) \
                for i in range(self.n_reductions))

    def _set_node_templates(self, template:Optional[ModelDesc])->None:
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

    def build_model_desc(self)->ModelDesc:
        # create model stems
        in_shapes = [[[self.ds_ch, -1, -1, -1]]]
        stem_out_shapes, model_stems = self.build_model_stems(in_shapes, self.conf_model_stems)

        # create cell descriptions
        cell_descs, aux_tower_descs = self.build_cells(in_shapes, model_stems,
           stem_reductions, self.max_final_edges)

        # get last cell channels
        if len(cell_descs):
            last_ch_out = cell_descs[-1].cell_ch_out
        else: # if no cells and only model stem
            last_ch_out = model_stems[-1].params['conv'].ch_out

        pool_op = self.get_pool_op(last_ch_out)

        logits_op = self.get_logits_op(last_ch_out)

        return ModelDesc(model_stems, pool_op, self.ds_ch,
                         self.n_classes, cell_descs, aux_tower_descs,
                         logits_op, self.model_desc_params.to_dict())

    def get_pool_op(self, ch_in:int)->OpDesc:
        return OpDesc(self.model_post_op,
                         params={'conv': ConvMacroParams(ch_in, ch_in)},
                         in_len=1, trainables=None)

    def get_logits_op(self, ch_in:int)->OpDesc:
        return OpDesc('linear',
                        params={'n_ch':ch_in,'n_classes': self.n_classes}, in_len=1,
                        trainables=None)

    def build_cell(self, in_shapes:TensorShapes, cell_conf:Config)\
            ->Tuple[TensorShapes, CellDesc, AuxTowerDesc]:

        p_ch_out = in_shapes[-1][0[0]
        pp_ch_out = in_shapes[-2][0][0] if len(in_shapes)>1 else p_ch_out
        is_reduction = self.is_reduction(ci)

        node_ch_out = 2*p_ch_out if is_reduction else p_ch_out

        reduction_p = in_shapes[0][0] > in_shapes[0][1] # previous output had reduction



    def build_cells(self, in_shapes:TensorShapes, cell_conf:Config)\
            ->Tuple[TensorShapes, List[CellDesc], List[Optional[AuxTowerDesc]]]:

        # TODO: support multiple stems
        assert len(in_shapes) == 1 and len(in_shapes[0])==2, "we must have only two stems output to start with"
        # both stems should have same channel outs (same true for cell outputs)
        assert in_shapes[-1][0][0] == in_shapes[-1][1][0]
        stem_ch_out = in_shapes[-1][0][0]

        cell_descs, aux_tower_descs = [], []

        # make copy of input shapes so we can keep adding more shapes as we add cells
        in_shapes = copy.deepcopy(in_shapes)
        for ci in range(self.n_cells):
            out_shapes, cell_desc, aux_desc = self.build_cell(in_shapes, ci, cell_conf)
            in_shapes.append(out_shapes)
            cell_descs.append(cell_desc)
            aux_tower_descs.append(aux_desc)

        # chennels of prev-prev whole cell, prev whole cell and current cell node
        pp_ch_out, p_ch_out, node_ch_out = stem_ch_out, stem_ch_out, self.init_node_ch

        # stores first cells of each time with whom arch params would be shared
        first_normal, first_reduction = -1, -1

        for ci in range(self.n_cells):
            # find cell type and output channels for this cell
            # also update if this is our first cell from which arch params will be shared
            reduction = self.is_reduction(ci)
            # if this is reduction cell then we will double the node channels in this cell
            if reduction:
                node_ch_out, cell_type = node_ch_out*2, CellType.Reduction
                first_reduction = ci if first_reduction < 0 else first_reduction
                template_cell = first_reduction
            else:
                cell_type = CellType.Regular
                first_normal = ci if first_normal < 0 else first_normal
                template_cell = first_normal

            s0_op, s1_op = self.get_cell_stems(
                node_ch_out, p_ch_out, pp_ch_out, reduction_p, ci)

            # cell template for this cell contains nodes we can use as template
            cell_template = self._cell_template(cell_type)

            # if cell template was not supplied then we
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
            # add nodes from the template to the just added cell
            self._copy_template_nodes(cell_template, cell_descs[ci])
            # add aux tower
            aux_tower_descs.append(self.get_aux_tower(cell_descs[ci], ci))

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

    def is_reduction(self, cell_index:int)->bool:
        # For darts, n_cells=8 so we build [N N R N N R N N] structure
        # Notice that this will result in only 2 reduction cells no matter
        # total number of cells. Original resnet actually have 3 reduction cells.
        # Between two reduction cells we have regular cells.
        return cell_index in self._reduction_indices

    def get_cell_stems(self, node_ch_out: int, p_ch_out: int, pp_ch_out:int,
                   reduction_p: bool, cell_index:int)->Tuple[OpDesc, OpDesc]:

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

    def get_aux_tower(self, cell_desc:CellDesc, cell_index:int)->Optional[AuxTowerDesc]:
        # TODO: shouldn't we be adding aux tower at *every* 1/3rd?
        if self.aux_weight and cell_index == 2*self.n_cells//3:
            return AuxTowerDesc(cell_desc.cell_ch_out, self.n_classes, self.aux_tower_stride)
        return None

    def build_model_stems(self, in_shapes:TensorShapes,
            conf_model_stems:Config)->Tuple[TensorShapes, List[OpDesc]]:
        # TODO: why do we need stem_multiplier?
        # TODO: in original paper stems are always affine

        init_node_ch = conf_model_stems['init_node_ch']
        stem_multiplier = conf_model_stems['stem_multiplier']
        ops = conf_model_stems['ops']

        conv_params = ConvMacroParams(in_shapes[-1][0][0], # channels of first input tensor
                                      init_node_ch*stem_multiplier)

        stems = [OpDesc(name=op_name, params={'conv': conv_params},
                          in_len=1, trainables=None) \
                for op_name in ops]

        # get reduction factors  done by each stem, typically they should be same but for
        # imagenet they can differ
        stem_reductions = ModelDescBuilder._stem_reductions(stems)

        out_shapes = [[init_node_ch, -1.0/stem_reduction, -1.0/stem_reduction, -1] for stem_reduction in stem_reductions]

        return out_shapes, stems

    @staticmethod
    def _stem_reductions(stems:List[OpDesc])->List[int]:
        #create stem ops to find out reduction factors
        ops = [Op.create(stem, affine=False) for stem in stems]
        assert all(isinstance(op, StemBase) for op in ops)
        return list(op.reduction for op in ops)