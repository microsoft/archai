# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from typing import Mapping, Optional, List, Tuple, Union
import pathlib
import os
import torch
import copy

import yaml

from archai.common import utils
from archai.common.common import logger

"""
Note: All classes in this file needs to be deepcopy compatible because
      descs are used as template to create copies by macro builder.
"""

class ConvMacroParams:
    """Holds parameters that may be altered by macro architecture"""

    def __init__(self, ch_in:int, ch_out:int) -> None:
        self.ch_in, self.ch_out = ch_in, ch_out

    def clone(self)->'ConvMacroParams':
        return copy.deepcopy(self)

class OpDesc:
    """Op description that is in each edge
    """
    def __init__(self, name:str, params:dict, in_len:int,
                 trainables:Optional[Mapping],
                 children:Optional[List['OpDesc']]=None,
                 children_ins:Optional[List[int]]=None)->None:
        self.name = name
        self.in_len = in_len
        self.params = params # parameters specific to op needed to construct it
        self.trainables = trainables # TODO: make this private due to clear_trainable
        # If op is keeping any child op then it should save it in children.
        # This way we can control state_dict of children.
        self.children = children
        self.children_ins = children_ins

    def clone(self, clone_trainables=True)->'OpDesc':
        cloned = copy.deepcopy(self)
        if not clone_trainables:
            cloned.clear_trainables()
        return cloned

    def clear_trainables(self)->None:
        self.trainables = None
        if self.children is not None:
            for child in self.children:
                child.clear_trainables()

    def state_dict(self)->dict:
        return  {
                    'trainables': self.trainables,
                    'children': [child.state_dict() if child is not None else None
                                 for child in self.children] \
                                     if self.children is not None else None
                }

    def load_state_dict(self, state_dict)->None:
        self.trainables = state_dict['trainables']
        c, cs = self.children, state_dict['children']
        assert (c is None and cs is None) or \
                (c is not None and cs is not None and len(c) == len(cs))
        # TODO: when c and cs are both none, zip throws an error that the 
        # first argument should be iterable
        if (c is None and cs is None):
            return 
        for cx, csx in zip(c, cs):
            if cx is not None and csx is not None:
                cx.load_state_dict(csx)


class EdgeDesc:
    """Edge description between two nodes in the cell
    """
    def __init__(self, op_desc:OpDesc, input_ids:List[int])->None:
        assert op_desc.in_len == len(input_ids)
        self.op_desc = op_desc
        self.input_ids = input_ids

    def clone(self, conv_params:Optional[ConvMacroParams], clear_trainables:bool)\
            ->'EdgeDesc':
        # edge cloning is same as deep copy except that we do it through
        # constructor for future proofing any additional future rules and
        # that we allow overiding conv_params and clearing weights
        e = EdgeDesc(self.op_desc.clone(), self.input_ids)
        # op_desc should have params set from cloning. If no override supplied
        # then don't change it
        if conv_params is not None:
            e.op_desc.params['conv'] = conv_params
        if clear_trainables:
            e.op_desc.clear_trainables()
        return e

    def clear_trainables(self)->None:
        self.op_desc.clear_trainables()

    def state_dict(self)->dict:
        return  {'op_desc': self.op_desc.state_dict()}

    def load_state_dict(self, state_dict)->None:
        self.op_desc.load_state_dict(state_dict['op_desc'])

class NodeDesc:
    def __init__(self, edges:List[EdgeDesc]) -> None:
        self.edges = edges

    def clone(self):
        # don't override conv_params or reset learned weights
        # node cloning is currently equivalent to deep copy
        return NodeDesc(edges=[e.clone(conv_params=None, clear_trainables=False)
                               for e in self.edges])

    def clear_trainables(self)->None:
        for edge in self.edges:
            edge.clear_trainables()


    def state_dict(self)->dict:
        return  { 'edges': [e.state_dict() for e in self.edges] }

    def load_state_dict(self, state_dict)->None:
        for e, es in zip(self.edges, state_dict['edges']):
            e.load_state_dict(es)

class AuxTowerDesc:
    def __init__(self, ch_in:int, n_classes:int, stride:int) -> None:
        self.ch_in = ch_in
        self.n_classes = n_classes
        self.stride = stride

class CellType(Enum):
    Regular = 'regular'
    Reduction  = 'reduction'

class CellDesc:
    def __init__(self, cell_type:CellType, id:int, nodes:List[NodeDesc],
            s0_op:OpDesc, s1_op:OpDesc, template_cell:int, max_final_edges:int,
            node_ch_out:int, post_op:Union[str,OpDesc])->None:
        assert s0_op.params['conv'].ch_out == s1_op.params['conv'].ch_out
        assert s0_op.params['conv'].ch_out == node_ch_out

        self.cell_type = cell_type
        self.id = id
        self.s0_op, self.s1_op = s0_op, s1_op
        self.template_cell = template_cell # cell id with which we share arch params
        self.max_final_edges = max_final_edges

        self.cell_ch_out = -1 # will be set later by reset_nodes
        self.reset_nodes(nodes, node_ch_out, post_op)
        assert self.cell_ch_out > 0

    def clone(self, id:int)->'CellDesc':
        c = copy.deepcopy(self) # note that template_cell is also cloned
        c.id = id
        return c

    def clear_trainables(self)->None:
        for attr in ['s0_op', 's1_op', 'post_op']:
            op_desc:OpDesc = getattr(self, attr)
            op_desc.clear_trainables()
        for node in self._nodes:
            node.clear_trainables()

    def nodes_editable(self)->bool:
        """Can we change node count without having to rebuild entire model desc?
           This is possible if post op outputs same number of channels regardless
           of node count.
        """
        _, cell_ch_out1, _ = CellDesc._post_op_ch(1, 1, self.post_op.name)
        _, cell_ch_out2, _ = CellDesc._post_op_ch(2, 1, self.post_op.name)
        return cell_ch_out1 == cell_ch_out2

    def state_dict(self)->dict:
        return  {
                    'nodes': [n.state_dict() for n in self.nodes()],
                    's0_op': self.s0_op.state_dict(),
                    's1_op': self.s1_op.state_dict(),
                    'post_op': self.post_op.state_dict()
                }

    def load_state_dict(self, state_dict)->None:
        for n, ns in zip(self.nodes(), state_dict['nodes']):
            n.load_state_dict(ns)
        self.s0_op.load_state_dict(state_dict['s0_op'])
        self.s1_op.load_state_dict(state_dict['s1_op'])
        self.post_op.load_state_dict(state_dict['post_op'])

    @staticmethod
    def _post_op_ch(node_count:int, node_ch_out:int,
                    post_op_name:str)->Tuple[int, int, int]:

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

    @staticmethod
    def create_post_op(node_count:int, node_ch_out:int,
                    post_op_name:str)->OpDesc:
        op_ch_in, cell_ch_out, out_states = CellDesc._post_op_ch(node_count,
                                                    node_ch_out, post_op_name)
        return OpDesc(post_op_name,
            {
                'conv': ConvMacroParams(op_ch_in, cell_ch_out),
                'out_states': out_states
            },
            in_len=1, trainables=None, children=None)

    def reset_nodes(self, nodes:List[NodeDesc], node_ch_out:int,
                    post_op:Union[str,OpDesc])->None:

        self._nodes = nodes
        self.node_ch_out = node_ch_out

        # we need to accept str as well as obj because during init we have name
        # but during finalize we have finalized post op object
        if isinstance(post_op, str):
            post_op = CellDesc.create_post_op(len(nodes), node_ch_out, post_op)

        post_op_ch_in = post_op.params['conv'].ch_in
        post_op_ch_out = post_op.params['conv'].ch_out
        post_op_out_states = post_op.params['out_states']

        # verify that the supplied op has channels as we expect
        post_op_ch_in_, post_op_ch_out_, post_op_out_states_ = \
            CellDesc._post_op_ch(len(nodes), node_ch_out, post_op.name)
        assert post_op_ch_in == post_op_ch_in_ and \
               post_op_ch_out == post_op_ch_out_ and \
               post_op_out_states == post_op_out_states_

        if self.cell_ch_out > -1 and self.cell_ch_out != post_op_ch_out:
            raise RuntimeError('Output channel of a cell cannot be resetted'
                               ' because this requires that all subsequent cells'
                               ' change to and model should be reconstructed.'
                               f' new cell_ch_out={post_op_ch_out},'
                               f' self.cell_ch_out={self.cell_ch_out}')

        self.post_op = post_op
        self.cell_ch_out = post_op_ch_out
        # conv parameters for each node
        self.conv_params = ConvMacroParams(node_ch_out, node_ch_out)
        # make sure we have real channel count
        assert self.cell_ch_out > -1

    def nodes(self)->List[NodeDesc]:
        return self._nodes

    def all_empty(self)->bool:
        return len(self._nodes)==0 or all((len(n.edges)==0 for n in self._nodes))
    def all_full(self)->bool:
        return len(self._nodes)>0 and all((len(n.edges)>0 for n in self._nodes))


class ModelDesc:
    def __init__(self, stem0_op:OpDesc, stem1_op:OpDesc, pool_op:OpDesc,
                 ds_ch:int, n_classes:int, cell_descs:List[CellDesc],
                 aux_tower_descs:List[Optional[AuxTowerDesc]],
                 logits_op:OpDesc, params:dict)->None:
        self.stem0_op, self.stem1_op, self.pool_op = stem0_op, stem1_op, pool_op
        self.logits_op = logits_op
        self.params = params

        self.ds_ch = ds_ch
        self.n_classes = n_classes

        self.reset_cells(cell_descs, aux_tower_descs)

    def reset_cells(self, cell_descs:List[CellDesc],
                    aux_tower_descs:List[Optional[AuxTowerDesc]])->None:
        assert len(cell_descs) == len(aux_tower_descs)
        # every cell should have unique ID so we can tell where arch params are shared
        assert len(set(c.id for c in cell_descs)) == len(cell_descs)

        self._cell_descs = cell_descs
        self.aux_tower_descs = aux_tower_descs

    def clear_trainables(self)->None:
        for attr in ['stem0_op', 'stem1_op', 'pool_op', 'logits_op']:
            op_desc:OpDesc = getattr(self, attr)
            op_desc.clear_trainables()
        for cell_desc in self._cell_descs:
            cell_desc.clear_trainables()

    def cell_descs(self)->List[CellDesc]:
        return self._cell_descs

    def cell_type_count(self, cell_type:CellType)->int:
        return sum(1 for c in self._cell_descs if c.cell_type==cell_type)

    def clone(self)->'ModelDesc':
        return copy.deepcopy(self)

    def has_aux_tower(self)->bool:
        return any(self.aux_tower_descs)

    def all_empty(self)->bool:
        return len(self._cell_descs)==0 or \
             all((c.all_empty() for c in self._cell_descs))
    def all_full(self)->bool:
        return len(self._cell_descs)>0 and \
            all((c.all_full() for c in self._cell_descs))
    def all_nodes_editable(self)->bool:
        return all((c.nodes_editable() for c in self._cell_descs))

    def state_dict(self)->dict:
        return  {
                    'cell_descs': [c.state_dict() for c in self.cell_descs()],
                    'stem0_op': self.stem0_op.state_dict(),
                    'stem1_op': self.stem1_op.state_dict(),
                    'pool_op': self.pool_op.state_dict(),
                    'logits_op': self.logits_op.state_dict()
                }

    def load_state_dict(self, state_dict)->None:
        for c, cs in zip(self.cell_descs(), state_dict['cell_descs']):
            c.load_state_dict(cs)
        self.stem0_op.load_state_dict(state_dict['stem0_op'])
        self.stem1_op.load_state_dict(state_dict['stem1_op'])
        self.pool_op.load_state_dict(state_dict['pool_op'])
        self.logits_op.load_state_dict(state_dict['logits_op'])

    def save(self, filename:str, save_trainables=False)->Optional[str]:
        if filename:
            filename = utils.full_path(filename)

            if save_trainables:
                state_dict = self.state_dict()
                pt_filepath = ModelDesc._pt_filepath(filename)
                torch.save(state_dict, pt_filepath)

            # save yaml
            cloned = self.clone()
            cloned.clear_trainables()
            pathlib.Path(filename).write_text(yaml.dump(cloned))

        return filename

    @staticmethod
    def _pt_filepath(desc_filepath:str)->str:
        # change file extension
        return str(pathlib.Path(desc_filepath).with_suffix('.pth'))

    @staticmethod
    def load(filename:str, load_trainables=False)->'ModelDesc':
        filename = utils.full_path(filename)
        if not filename or not os.path.exists(filename):
            raise RuntimeError("Model description file is not found."
                "Typically this file should be generated from the search."
                "Please copy this file to '{}'".format(filename))

        logger.info({'final_desc_filename': filename})
        with open(filename, 'r') as f:
            model_desc = yaml.load(f, Loader=yaml.Loader)

        if load_trainables:
            # look for pth file that should have pytorch parameters state_dict
            pt_filepath = ModelDesc._pt_filepath(filename)
            if os.path.exists(pt_filepath):
                state_dict = torch.load(pt_filepath, map_location=torch.device('cpu'))
                model_desc.load_state_dict(state_dict)
            # else no need to restore weights

        return model_desc
