# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Collection, Optional, Tuple, List
import copy

from overrides import EnforceOverrides

from archai.common.config import Config
from archai.nas.model_desc import ModelDesc, OpDesc, CellType, NodeDesc, EdgeDesc, \
                                  CellDesc, AuxTowerDesc, ConvMacroParams, \
                                  TensorShape, TensorShapes, TensorShapesList
from archai.common.common import logger
from archai.nas.operations import StemBase, Op


class ModelDescBuilder(EnforceOverrides):
    def get_reduction_indices(self, conf_model_desc:Config)->List[int]:
        """ Returns cell indices which reduces HxW and doubles channels """

        n_cells:int = conf_model_desc['n_cells']
        n_reductions:int = conf_model_desc['n_reductions']

        # this satisfies N R N R N pattern, this need not be enforced but
        # we are doing now for sanity
        assert n_cells >= n_reductions * 2 + 1

        # for each reduction, we create one indice
        # for cifar and imagenet, reductions=2 creating cuts at n//3, n*2//3
        return list(n_cells*(i+1) // (n_reductions+1) \
                for i in range(n_reductions))

    def get_node_channels(self, conf_model_desc:Config)->List[List[int]]:
        """ Returns array of channels for each node in each cell. All nodes
            are assumed to have same output channels as input channels. """

        conf_model_stems = self.get_conf_model_stems()
        conf_cell = self.get_conf_cell()

        init_node_ch:int = conf_model_stems['init_node_ch']
        n_cells = conf_model_desc['n_cells']
        n_nodes = conf_cell['n_nodes']

        # same channels for all nodes in a cell
        cell_node_channels:List[List[int]] = []
        # channels for the first cell
        node_channels = init_node_ch

        for ci in range(n_cells):
            # if reduction cell than double the node channels
            if self.get_cell_type(ci)==CellType.Reduction:
                node_channels *= 2
            # all nodes in a cell have same channels
            nodes_channels = [node_channels for ni in range(n_nodes)]
            cell_node_channels.append(nodes_channels
                                      )
        return cell_node_channels

    def get_conf_cell(self)->Config:
        return self.conf_model_desc['cell']

    def get_conf_dataset(self)->Config:
        return self.conf_model_desc['dataset']

    def get_conf_model_stems(self)->Config:
        return self.conf_model_desc['model_stems']

    def _init_build(self, conf_model_desc: Config,
                 template:Optional[ModelDesc]=None)->None:

        self.conf_model_desc = conf_model_desc
        self.template = template
        # if template model desc is specified then setup regular and reduction cell templates
        self._cell_templates = self.create_cell_templates(template)

        n_cells = conf_model_desc['n_cells']

        # for each reduction, we create one indice
        # for cifar and imagenet, reductions=2 creating cuts at n//3, n*2//3
        self._reduction_indices = self.get_reduction_indices(conf_model_desc)
        self._normal_indices = [i for i in range(n_cells)\
                                if i not in self._reduction_indices]
        self.node_channels = self.get_node_channels(conf_model_desc)

    def build(self, conf_model_desc: Config,
                 template:Optional[ModelDesc]=None)->ModelDesc:
        """main entry point for the class"""

        self._init_build(conf_model_desc, template)

        self.pre_build(conf_model_desc)

        # input shape for the stem has same channels as channels in image
        # -1 indicates, actual dimensions are not known
        ds_ch = self.get_conf_dataset()['channels']
        in_shapes = [[[ds_ch, -1, -1, -1]]]

        # create model stems
        model_stems = self.build_model_stems(in_shapes, conf_model_desc)

        # create cell descriptions
        cell_descs, aux_tower_descs = self.build_cells(in_shapes, conf_model_desc)

        model_pool_op = self.build_model_pool(in_shapes, conf_model_desc)

        logits_op = self.build_logits_op(in_shapes, conf_model_desc)

        return ModelDesc(conf_model_desc, model_stems, model_pool_op, cell_descs,
                         aux_tower_descs, logits_op)

    def build_cells(self, in_shapes:TensorShapesList, conf_model_desc:Config)\
            ->Tuple[List[CellDesc], List[Optional[AuxTowerDesc]]]:

        conf_cell = self.get_conf_cell()

        n_cells = conf_model_desc['n_cells']

        cell_descs, aux_tower_descs = [], []

        # create list of output shapes for cells that starts with model stem
        for ci in range(n_cells):
            cell_desc = self.build_cell(in_shapes, conf_cell, ci)
            # get first tensor output of last cell
            aux_tower_desc = self.build_aux_tower(in_shapes[-1][0], conf_model_desc, ci)
            cell_descs.append(cell_desc)
            aux_tower_descs.append(aux_tower_desc)

        return cell_descs, aux_tower_descs

    def get_node_count(self, cell_index:int)->int:
        return len(self.node_channels[cell_index])

    def build_cell(self, in_shapes:TensorShapesList, conf_cell:Config,
                   cell_index:int) ->CellDesc:

        stem_shapes, stems = self.build_cell_stems(in_shapes, conf_cell, cell_index)
        cell_type = self.get_cell_type(cell_index)

        if self.template is None:
            node_count = self.get_node_count(cell_index)
            in_shape = stem_shapes[0] # input shape to noded is same as cell stem
            out_shape = stem_shapes[0] # we ask nodes to keep the output shape same
            node_shapes, nodes = self.build_nodes(stem_shapes, conf_cell,
                                                  cell_index, cell_type, node_count, in_shape, out_shape)
        else:
            node_shapes, nodes = self.build_nodes_from_template(stem_shapes, conf_cell, cell_index)

        post_op_shape, post_op_desc = self.build_cell_post_op(stem_shapes,
            node_shapes, conf_cell, cell_index)

        cell_desc = CellDesc(
            id=cell_index, cell_type=self.get_cell_type(cell_index),
            conf_cell=conf_cell,
            stems=stems, stem_shapes=stem_shapes,
            nodes=nodes, node_shapes=node_shapes,
            post_op=post_op_desc, out_shape=post_op_shape,
            trainables_from=self.get_trainables_from(cell_index)
        )

        # output same shape twice to indicate s0 and s1 inputs for next cell
        in_shapes.append([post_op_shape])

        return cell_desc

    def get_trainables_from(self, cell_index:int)->int:
        cell_type = self.get_cell_type(cell_index)
        if cell_type == CellType.Reduction:
            return self._reduction_indices[0]
        if cell_type == CellType.Regular:
            return self._normal_indices[0]
        raise RuntimeError(f'Cannot get cell for shared trainables because cell_type "{cell_type}" is not recgnized')

    def get_ch(self, shape:TensorShape)->int:
        return int(shape[0])

    def build_cell_stems(self, in_shapes:TensorShapesList, conf_cell:Config,
                   cell_index:int)\
                       ->Tuple[TensorShapes, List[OpDesc]]:

        # expect two stems, both should have same channels
        # TODO: support multiple stems
        assert len(in_shapes) >= 2, "we must have outputs from at least two previous modules"

        # Get channels for previous two layers.
        # At start we have only one layer, i.e., model stems.
        # Typically model stems should have same channel count but for imagenet we do
        #   reduction at model stem so stem1 will have twice channels as stem0
        p_ch_out = self.get_ch(in_shapes[-1][0])
        pp_ch_out = self.get_ch(in_shapes[-2][0])

        # was the previous layer reduction layer?
        reduction_p = p_ch_out == pp_ch_out*2 or in_shapes[-2][0][2] == in_shapes[-1][0][2]*2

        # find out the node channels for this cell
        node_ch_out = self.node_channels[cell_index][0] # init with first node in cell

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

        # output two shapes with proper channels setup
        # for default model desc, cell stems have same shapes and channels
        out_shape0 = copy.deepcopy(in_shapes[-1][0])
        # set channels and reset shapes to -1 to indicate unknown
        # for imagenet HxW would be floating point numbers due to one input reduced
        out_shape0[0], out_shape0[2], out_shape0[3] = node_ch_out, -1, -1
        out_shape1 = copy.deepcopy(out_shape0)

        return [out_shape0, out_shape1], [s0_op, s1_op]

    def build_nodes_from_template(self, stem_shapes:TensorShapes, conf_cell:Config,
                    cell_index:int) \
                        ->Tuple[TensorShapes, List[NodeDesc]]:

        cell_template = self.get_cell_template(cell_index)

        assert cell_template is not None

        cell_type = self.get_cell_type(cell_index)
        assert cell_template.cell_type==cell_type

        nodes:List[NodeDesc] = []
        for n in cell_template.nodes():
            edges_copy = [e.clone(
                            # use new macro params
                            conv_params=ConvMacroParams(self.get_ch(stem_shapes[0]),
                                                        self.get_ch(stem_shapes[0])),
                            # TODO: check for compatibility?
                            clear_trainables=True
                            ) for e in n.edges]
            nodes.append(NodeDesc(edges=edges_copy, conv_params=n.conv_params))

        out_shapes = [copy.deepcopy(stem_shapes[0]) for _  in cell_template.nodes()]

        return out_shapes, nodes

    def build_nodes(self, stem_shapes:TensorShapes, conf_cell:Config,
                    cell_index:int, cell_type:CellType, node_count:int,
                    in_shape:TensorShape, out_shape:TensorShape) \
                        ->Tuple[TensorShapes, List[NodeDesc]]:

        # default: create nodes with empty edges
        nodes:List[NodeDesc] =  [NodeDesc(edges=[],
                                            conv_params=ConvMacroParams(
                                                self.get_ch(in_shape),
                                                self.get_ch(out_shape)))
                                for _ in range(node_count)]

        out_shapes = [copy.deepcopy(out_shape) for _  in range(node_count)]

        return out_shapes, nodes

    def create_cell_templates(self, template:Optional[ModelDesc])\
            ->List[Optional[CellDesc]]:
        normal_template,  reduction_template = None, None
        if template is not None:
            # find first regular and reduction cells and set them as
            # the template that we will use. When we create new cells
            # we will fill them up with nodes from these templates
            for cell_desc in template.cell_descs():
                if normal_template is None and \
                        cell_desc.cell_type==CellType.Regular:
                  normal_template = cell_desc
                if reduction_template is None and \
                        cell_desc.cell_type==CellType.Reduction:
                    reduction_template = cell_desc
        return [normal_template, reduction_template]

    def build_model_pool(self, in_shapes:TensorShapesList, conf_model_desc:Config)\
            ->OpDesc:

        model_post_op = conf_model_desc['model_post_op']
        last_shape = in_shapes[-1][0]

        in_shapes.append([copy.deepcopy(last_shape)])

        return OpDesc(model_post_op,
                         params={'conv': ConvMacroParams(self.get_ch(last_shape),
                                                         self.get_ch(last_shape))},
                         in_len=1, trainables=None)

    def build_logits_op(self, in_shapes:TensorShapesList, conf_model_desc:Config)->OpDesc:
        n_classes = self.get_conf_dataset()['n_classes']

        return OpDesc('linear',
                        params={'n_ch':in_shapes[-1][0][0],
                                'n_classes': n_classes},
                        in_len=1, trainables=None)

    def get_cell_template(self, cell_index:int)->Optional[CellDesc]:
        cell_type = self.get_cell_type(cell_index)

        if cell_type==CellType.Regular:
            return self._cell_templates[0]
        if cell_type==CellType.Reduction:
            return self._cell_templates[1]
        raise RuntimeError(f'Cannot get cell template because cell_type "{cell_type}" is not recgnized')

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
        node_ch_out = self.get_ch(node_shapes[-1])

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
            node_shapes:TensorShapes, conf_cell:Config, cell_index:int)\
                -> Tuple[TensorShape, OpDesc]:

        post_op_name = conf_cell['cell_post_op']
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

    def build_aux_tower(self, out_shape:TensorShape, conf_model_desc:Config,
                        cell_index:int)->Optional[AuxTowerDesc]:
        n_classes = self.get_conf_dataset()['n_classes']
        n_cells = conf_model_desc['n_cells']
        n_reductions = conf_model_desc['n_reductions']
        aux_tower_stride = conf_model_desc['aux_tower_stride']
        aux_weight = conf_model_desc['aux_weight']

        # TODO: shouldn't we be adding aux tower at *every* 1/3rd?
        if aux_weight and n_reductions > 1 and cell_index == 2*n_cells//3:
            return AuxTowerDesc(self.get_ch(out_shape), n_classes, aux_tower_stride)
        return None

    def build_model_stems(self, in_shapes:TensorShapesList,
            conf_model_desc:Config)->List[OpDesc]:
        # TODO: why do we need stem_multiplier?
        # TODO: in original paper stems are always affine

        conf_model_stems = self.get_conf_model_stems()

        init_node_ch:int = conf_model_stems['init_node_ch']
        stem_multiplier:int = conf_model_stems['stem_multiplier']
        ops:List[str] = conf_model_stems['ops']

        out_channels = init_node_ch*stem_multiplier

        conv_params = ConvMacroParams(self.get_ch(in_shapes[-1][0]), # channels of first input tensor
                                      init_node_ch*stem_multiplier)

        stems = [OpDesc(name=op_name, params={'conv': conv_params},
                          in_len=1, trainables=None) \
                for op_name in ops]

        # get reduction factors  done by each stem, typically they should be same but for
        # imagenet they can differ
        stem_reductions = ModelDescBuilder._stem_reductions(stems)

        # Each cell takes input from previous and 2nd previous cells.
        # To be consistence we create two outputs for model stems: [[s1, s0], [s0, s1]
        # This way when we access first element of each output we get s1, s0.
        # Normailly s0==s1 but for networks like imagenet, s0 will have twice the channels
        #   of s1.
        for stem_reduction in stem_reductions:
            in_shapes.append([[out_channels, -1, -1.0/stem_reduction, -1.0/stem_reduction]])

        return stems

    @staticmethod
    def _stem_reductions(stems:List[OpDesc])->List[int]:
        # create stem ops to find out reduction factors
        ops = [Op.create(stem, affine=False) for stem in stems]
        assert all(isinstance(op, StemBase) for op in ops)
        return list(op.reduction for op in ops)

    def pre_build(self, conf_model_desc:Config)->None:
        """hook for accomplishing any setup before build starts"""
        pass

    def seed_cell(self, model_desc:ModelDesc)->None:
        # prepare model as seed model before search iterations starts
        pass
