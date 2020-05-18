from enum import Enum
from typing import Mapping, Optional, List, Tuple, Union
import pathlib
import os
import torch
import copy

import yaml

from ..common.common import expdir_abspath, expdir_filepath


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
        c_children = None if self.children is None else [c.clone() for c in self.children]
        return OpDesc(self.name,
                      copy.deepcopy(self.params), # because these may contain objects!
                      self.in_len,
                      None if not clone_trainables else copy.deepcopy(self.trainables),
                      c_children,
                      copy.deepcopy(self.children_ins) # don't leak by keeping refs
                      )

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
                                     if self.children is not None else None,
                    'children_ins': [child_in if child_in is not None else None
                                 for child_in in self.children_ins] \
                                     if self.children_ins is not None else None,
                }

    def load_state_dict(self, state_dict)->None:
        if state_dict is not None:
            self.trainables = state_dict['trainables']
            c, cs = self.children, state_dict['children']
            assert (c is None and cs is None) or \
                   (c is not None and cs is not None and len(c) == len(cs))
            c_ins, cs_ins = self.children_ins, state_dict['children_ins']
            assert (c_ins is None and cs_ins is None) or \
                   (c_ins is not None and cs_ins is not None and len(c_ins) == len(cs_ins))
            if c is not None:
                assert c_ins is not None and len(c) == len(c_ins)
                for i, (cx, csx, csix) in enumerate(zip(c, cs, cs_ins)):
                    assert (cx is None and csx is None) or \
                            (cx is not None and csx is not None)
                    if cx is not None:
                        assert csix is not None
                        cx.load_state_dict(csx)
                        c_ins[i] = csix

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
        # that we allow oveeriding conv_params and clearning weights

        e = EdgeDesc(self.op_desc.clone(), self.input_ids)
        # op_desc should have params set from cloning. If no override supplied
        # then don't change it
        if conv_params is not None:
            e.op_desc.params['conv'] = conv_params
        if clear_trainables:
            e.op_desc.clear_trainables()
        return e

class NodeDesc:
    def __init__(self, edges:List[EdgeDesc]) -> None:
        self.edges = edges

    def clone(self):
        # don't override conv_params or reset learned weights
        # node cloning is currently equivalent to deep copy
        return NodeDesc(edges=[e.clone(conv_params=None, clear_trainables=False)
                               for e in self.edges])

class AuxTowerDesc:
    def __init__(self, ch_in:int, n_classes:int) -> None:
        self.ch_in = ch_in
        self.n_classes = n_classes

class CellType(Enum):
    Regular = 'regular'
    Reduction  = 'reduction'

class CellDesc:
    def __init__(self, cell_type:CellType, id:int, nodes:List[NodeDesc],
            s0_op:OpDesc, s1_op:OpDesc, alphas_from:int, max_final_edges:int,
            node_ch_out:int, post_op:Union[str,OpDesc])->None:
        assert s0_op.params['conv'].ch_out == s1_op.params['conv'].ch_out
        assert s0_op.params['conv'].ch_out == node_ch_out

        self.cell_type = cell_type
        self.id = id
        self.s0_op, self.s1_op = s0_op, s1_op
        self.alphas_from = alphas_from # cell id with which we share alphas
        self.max_final_edges = max_final_edges

        self.cell_ch_out = -1 # will be set by reset_nodes
        self.reset_nodes(nodes, node_ch_out, post_op)
        assert self.cell_ch_out > 0

    def clone(self, id:int)->'CellDesc':
        c = copy.deepcopy(self) # note that alphas_from is also cloned
        c.id = id
        return c

    def nodes_editable(self)->bool:
        """Can we change node count without having to rebuild entire model desc?
           This is possible if post op outputs same number of channels regardless
           of node count.
        """
        _, cell_ch_out1, _ = CellDesc._post_op_ch(1, 1, self.post_op.name)
        _, cell_ch_out2, _ = CellDesc._post_op_ch(2, 1, self.post_op.name)
        return cell_ch_out1 == cell_ch_out2

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
        # every cell should have unique ID so we can tell where alphas are shared
        assert len(set(c.id for c in cell_descs)) == len(cell_descs)

        self._cell_descs = cell_descs
        self.aux_tower_descs = aux_tower_descs

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

    def state_dict(self, clear=False)->dict:
        cells_state_dict = {}
        for ci, cell_desc in enumerate(self._cell_descs):
            sd_cell = cells_state_dict[ci] = {}
            for ni, node in enumerate(cell_desc.nodes()):
                sd_node = sd_cell[ni] = {}
                for ei, edge_desc in enumerate(node.edges):
                    sd_node[ei] = edge_desc.op_desc.state_dict()
                    if clear:
                        edge_desc.op_desc.clear_trainables()
        attrs_state_dict = {}
        for attr in ['stem0_op', 'stem1_op', 'pool_op', 'logits_op']:
            op_desc = getattr(self, attr)
            attrs_state_dict[attr] = op_desc.state_dict()
            if clear:
                op_desc.clear_trainables()
        return {'cells': cells_state_dict, 'attr': attrs_state_dict}

    def load_state_dict(self, state_dict:dict)->None:
        cells_state_dict = state_dict['cells']
        attrs_state_dict = state_dict['attr']
        # restore cells
        for ci, cell_desc in enumerate(self._cell_descs):
            sd_cell = cells_state_dict[ci]
            for ni, node in enumerate(cell_desc.nodes()):
                sd_node = sd_cell[ni]
                for ei, edge_desc in enumerate(node.edges):
                    edge_desc.op_desc.load_state_dict(sd_node[ei])

        # restore attributes
        for attr, attr_state_dict in attrs_state_dict.items():
            op_desc = getattr(self, attr)
            op_desc.load_state_dict(attr_state_dict)

    def save(self, filename:str, subdir:List[str]=[])->Optional[str]:
        yaml_filepath = expdir_filepath(filename, subdir)
        if yaml_filepath:
            if not yaml_filepath.endswith('.yaml'):
                yaml_filepath += '.yaml'

            # clear so PyTorch state is not saved in yaml
            state_dict = self.state_dict(clear=True)
            pt_filepath = ModelDesc._pt_filepath(yaml_filepath)
            torch.save(state_dict, pt_filepath)
            # save yaml
            pathlib.Path(yaml_filepath).write_text(yaml.dump(self))
            # restore state
            self.load_state_dict(state_dict)

        return yaml_filepath

    @staticmethod
    def _pt_filepath(desc_filepath:str)->str:
        # change file extension
        return str(pathlib.Path(desc_filepath).with_suffix('.pth'))

    @staticmethod
    def load(yaml_filename:str)->'ModelDesc':
        yaml_filepath = expdir_abspath(yaml_filename)
        if not yaml_filepath or not os.path.exists(yaml_filepath):
            raise RuntimeError("Model description file is not found."
                "Typically this file should be generated from the search."
                "Please copy this file to '{}'".format(yaml_filepath))
        with open(yaml_filepath, 'r') as f:
            model_desc = yaml.load(f, Loader=yaml.Loader)

        # look for pth file that should have pytorch parameters state_dict
        pt_filepath = ModelDesc._pt_filepath(yaml_filepath)
        if os.path.exists(pt_filepath):
            state_dict = torch.load(pt_filepath, map_location=torch.device('cpu'))
            model_desc.load_state_dict(state_dict)
        # else no need to restore weights
        return model_desc
