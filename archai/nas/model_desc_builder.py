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

# Each tensor shape is list
# A layer can output multiple tensors so its shapes are TensorShapes
# list of all layer outputs is TensorShapesList]
TensorShape=List[int]
TensorShapes=List[TensorShape]
TensorShapesList=List[TensorShapes]


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
        self.node_channels = self.get_node_channels()

        # if template model desc is specified thehn setup regular and reduction cell templates
        self._set_node_templates(template)

    def _get_reduction_indices(self)->set:
        # this satisfies N R N R N pattern
        assert self.n_cells >= self.n_reductions * 2 + 1

        # for each reduction, we create one indice
        # for cifar and imagenet, reductions=2 creating cuts at n//3, n*2//3
        return set(self.n_cells*(i+1) // (self.n_reductions+1) \
                for i in range(self.n_reductions))

    def get_node_channels(self)->List[List[int]]:
        cell_node_channels:List[List[int]] = []
        node_channels = self.init_node_ch
        for ci in range(self.n_cells):
            if ci in self._reduction_indices:
                node_channels *= node_channels
            nodes_channels = [node_channels for ni in range(self.n_nodes)]
            cell_node_channels.append(nodes_channels)
        return cell_node_channels


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

    def build_model_desc(self, cell_conf:Config)->ModelDesc:
        # create model stems
        in_shapes = [[[self.ds_ch, -1, -1, -1]]]
        stem_out_shapes, model_stems = self.build_model_stems(in_shapes, self.conf_model_stems)

        # create cell descriptions
        cell_out_shapes, cell_descs, aux_tower_descs = self.build_cells(in_shapes, cell_conf)

        pool_out_shape, model_pool_op = self.build_model_pool(cell_out_shapes)

        logits_op = self.build_logits_op(pool_out_shape)

        return ModelDesc(model_stems, model_pool_op, self.ds_ch,
                         self.n_classes, cell_descs, aux_tower_descs,
                         logits_op, self.model_desc_params.to_dict())

    def build_model_pool(self, out_shapes:TensorShapesList)\
            ->Tuple[TensorShape, OpDesc]:
        last_shape = out_shapes[-1][0]

        return copy.deepcopy(last_shape), OpDesc(self.model_post_op,
                         params={'conv': ConvMacroParams(last_shape[0], last_shape[0])},
                         in_len=1, trainables=None)

    def build_logits_op(self, out_shape:TensorShape)->OpDesc:
        return OpDesc('linear',
                        params={'n_ch':out_shape[0],'n_classes': self.n_classes}, in_len=1,
                        trainables=None)

    def build_cell(self, in_shapes:TensorShapesList, cell_conf:Config, ci:int)\
            ->Tuple[TensorShapes, CellDesc, Optional[AuxTowerDesc]]:

        cell_type = self.get_cell_type(ci)
        reduction = cell_type == CellType.Reduction

        stem_shapes, stems = self.build_cell_stems(in_shapes, cell_conf,
                                                   ci, reduction)

        node_shapes, nodes = self.build_nodes(stem_shapes, cell_conf, ci, reduction)

        post_op_shape, post_op_desc = self.build_cell_post_op(stem_shapes,
            node_shapes, cell_conf, ci, reduction)

        cell_desc = CellDesc(
            cell_type=cell_type, id=ci,
            nodes=nodes,
            stems=stems,
            post_op_desc=post_op_desc
        )

        aux_tower_desc = self.build_aux_tower(cell_desc, ci)

        return [post_op_shape], cell_desc, aux_tower_desc

    def build_cells(self, in_shapes:TensorShapesList, cell_conf:Config)\
            ->Tuple[TensorShapesList, List[CellDesc], List[Optional[AuxTowerDesc]]]:

        # TODO: support multiple stems
        assert len(in_shapes) == 1 and len(in_shapes[0])==2, "we must have only two stems output to start with"
        # both stems should have same channel outs (same true for cell outputs)
        assert in_shapes[-1][0][0] == in_shapes[-1][1][0]
        stem_ch_out = in_shapes[-1][0][0]

        cell_descs, aux_tower_descs = [], []

        # make copy of input shapes so we can keep adding more shapes as we add cells
        out_shapes = copy.deepcopy(in_shapes) # include model stem shapes in out shapes
        for ci in range(self.n_cells):
            out_shape, cell_desc, aux_desc = self.build_cell(out_shapes, cell_conf, ci)
            out_shapes.append(out_shape)
            cell_descs.append(cell_desc)
            aux_tower_descs.append(aux_desc)

        return out_shapes, cell_descs, aux_tower_descs

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

    def get_cell_type(self, cell_index:int)->CellType:
        # For darts, n_cells=8 so we build [N N R N N R N N] structure
        # Notice that this will result in only 2 reduction cells no matter
        # total number of cells. Original resnet actually have 3 reduction cells.
        # Between two reduction cells we have regular cells.
        return CellType.Reduction if cell_index in self._reduction_indices \
                                  else CellType.Regular

    def _post_op_ch(self, post_op_name:str, node_shapes:TensorShapes) \
            ->Tuple[int, int, int]:

        node_count = len(node_shapes)
        node_ch_out = node_shapes[-1][0]

        # we take all available node outputs as input to post op
        # if no nodes exist then we will use cell stem outputs
        # Note that for reduction cell stems wxh is larger than node wxh which
        # means we cannot use cell stem outputs with node outputs because
        # concate will fail
        # TODO: remove hard coding of 2
        out_states = node_count if node_count else 2

        # number of input channels to the cell post op
        op_ch_in = out_states * node_ch_out

        # number of output channels for the cell post op
        if post_op_name == 'concate_channels':
            cell_ch_out = op_ch_in
        elif post_op_name == 'proj_channels':
            cell_ch_out = node_ch_out
        else:
            raise RuntimeError(f'Unsupported cell_post_op: {post_op_name}')
        return op_ch_in, cell_ch_out, out_states

    def build_cell_post_op(self, stem_shapes:TensorShapes,
        node_shapes:TensorShapes, cell_conf:Config, cell_index:int,
        reduction:bool) -> Tuple[TensorShape, OpDesc]:

        post_op_name = self.cell_post_op
        op_ch_in, cell_ch_out, out_states = self._post_op_ch(post_op_name,
                                                             node_shapes)

        post_op_desc = OpDesc(post_op_name,
            {
                'conv': ConvMacroParams(op_ch_in, cell_ch_out),
                'out_states': out_states
            },
            in_len=1, trainables=None, children=None)

        out_shape = copy.deepcopy(node_shapes[-1])
        out_shape[0] = cell_ch_out

        return out_shape, post_op_desc

    def build_nodes(self, stem_shapes:TensorShapes, cell_conf:Config,
                    cell_index:int, reduction:bool) \
                        ->Tuple[TensorShapes, List[NodeDesc]]:

        node_count = len(self.node_channels[cell_index])

        # create nodes with empty edges
        nodes:List[NodeDesc] =  [NodeDesc(edges=[])
                                    for _ in range(node_count)]

        out_shapes = [copy.deepcopy(stem_shapes[0]) for _  in range(node_count)]

        return out_shapes, nodes

    def build_cell_stems(self, in_shapes:TensorShapesList, cell_conf:Config,
                   cell_index:int, reduction:bool)\
                       ->Tuple[TensorShapes, List[OpDesc]]:

        p_ch_out = in_shapes[-1][0][0]
        pp_ch_out = in_shapes[-2][0][0] if len(in_shapes)>1 else p_ch_out
        reduction_p = p_ch_out >= pp_ch_out*2
        node_ch_out = self.node_channels[cell_index][0] # first node in cell

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

        out_shape0 = copy.deepcopy(in_shapes[-1][0])
        out_shape0[0] = node_ch_out
        out_shape1 = copy.deepcopy(out_shape0)

        return [out_shape0, out_shape1], [s0_op, s1_op]

    def build_aux_tower(self, cell_desc:CellDesc, cell_index:int)->Optional[AuxTowerDesc]:
        # TODO: shouldn't we be adding aux tower at *every* 1/3rd?
        if self.aux_weight and cell_index == 2*self.n_cells//3:
            return AuxTowerDesc(cell_desc.cell_ch_out, self.n_classes, self.aux_tower_stride)
        return None

    def build_model_stems(self, in_shapes:TensorShapesList,
            conf_model_stems:Config)->Tuple[TensorShapes, List[OpDesc]]:
        # TODO: why do we need stem_multiplier?
        # TODO: in original paper stems are always affine

        init_node_ch:int = conf_model_stems['init_node_ch']
        stem_multiplier:int = conf_model_stems['stem_multiplier']
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