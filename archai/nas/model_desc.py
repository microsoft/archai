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
from archai.common.config import Config

"""
Note: All classes in this file needs to be deepcopy compatible because
      descs are used as template to create copies by macro builder.
"""

# Each tensor shape is list
# A layer can output multiple tensors so its shapes are TensorShapes
# list of all layer outputs is TensorShapesList]
TensorShape=List[Union[float]]
TensorShapes=List[TensorShape]
TensorShapesList=List[TensorShapes]

class ConvMacroParams:
    """Holds parameters that may be altered by macro architecture"""

    def __init__(self, ch_in:int, ch_out:int) -> None:
        self.ch_in, self.ch_out = ch_in, ch_out

    def clone(self)->'ConvMacroParams':
        return copy.deepcopy(self)

class OpDesc:
    """Op description that is in each edge"""

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
        for cx, csx in utils.zip_eq(c, cs):
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
    def __init__(self, edges:List[EdgeDesc], conv_params:ConvMacroParams) -> None:
        self.edges = edges
        self.conv_params = conv_params

    def clone(self):
        # don't override conv_params or reset learned weights
        # node cloning is currently equivalent to deep copy
        return NodeDesc(edges=[e.clone(conv_params=None, clear_trainables=False)
                               for e in self.edges], conv_params=self.conv_params)

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
    def __init__(self, id:int, cell_type:CellType, conf_cell:Config,
                 stems:List[OpDesc], stem_shapes:TensorShapes,
                 nodes:List[NodeDesc], node_shapes: TensorShapes,
            post_op:OpDesc, out_shape:TensorShape, trainables_from:int)->None:

        self.cell_type = cell_type
        self.id = id
        self.conf_cell = conf_cell
        self.stems = stems
        self.stem_shapes = stem_shapes
        self.out_shape = out_shape
        self.trainables_from = trainables_from
        self.reset_nodes(nodes, node_shapes, post_op, out_shape)

    def clone(self, id:int)->'CellDesc':
        c = copy.deepcopy(self) # note that trainables_from is also cloned
        c.id = id
        return c

    def clear_trainables(self)->None:
        for stem in self.stems:
            stem.clear_trainables()
        for node in self._nodes:
            node.clear_trainables()
        self.post_op.clear_trainables()

    def state_dict(self)->dict:
        return  {
                    'id': self.id,
                    'cell_type': self.cell_type,
                    'stems': [s.state_dict() for s in self.stems],
                    'stem_shapes': self.stem_shapes,
                    'nodes': [n.state_dict() for n in self.nodes()],
                    'node_shapes': self.node_shapes,
                    'post_op': self.post_op.state_dict(),
                    'out_shape': self.out_shape
                }

    def load_state_dict(self, state_dict)->None:
        assert self.id == state_dict['id']
        assert self.cell_type == state_dict['cell_type']

        for s, ss in utils.zip_eq(self.stems, state_dict['stems']):
            s.load_state_dict(ss)
        self.stem_shapes = state_dict['stem_shapes']

        for n, ns in utils.zip_eq(self.nodes(), state_dict['nodes']):
            n.load_state_dict(ns)
        self.node_shapes = state_dict['node_shapes']

        self.post_op.load_state_dict(state_dict['post_op'])
        self.out_shape = state_dict['out_shape']

    def reset_nodes(self, nodes:List[NodeDesc], node_shapes:TensorShapes,
                    post_op:OpDesc, out_shape:TensorShape)->None:
        self._nodes = nodes
        self.node_shapes = node_shapes
        self.post_op = post_op
        self.out_shape = out_shape

    def nodes(self)->List[NodeDesc]:
        return self._nodes

    def all_empty(self)->bool:
        return len(self._nodes)==0 or all((len(n.edges)==0 for n in self._nodes))
    def all_full(self)->bool:
        return len(self._nodes)>0 and all((len(n.edges)>0 for n in self._nodes))


class ModelDesc:
    def __init__(self, conf_model_desc:Config, model_stems:List[OpDesc], pool_op:OpDesc,
                 cell_descs:List[CellDesc], aux_tower_descs:List[Optional[AuxTowerDesc]],
                 logits_op:OpDesc)->None:

        self.conf_model_desc = conf_model_desc
        conf_dataset = conf_model_desc['dataset']
        self.ds_ch:int = conf_dataset['channels']
        self.n_classes:int = conf_dataset['n_classes']
        self.params = conf_model_desc['params'].to_dict()
        self.max_final_edges:int = conf_model_desc['max_final_edges']

        self.model_stems, self.pool_op = model_stems, pool_op
        self.logits_op = logits_op

        self.reset_cells(cell_descs, aux_tower_descs)

    def reset_cells(self, cell_descs:List[CellDesc],
                    aux_tower_descs:List[Optional[AuxTowerDesc]])->None:
        assert len(cell_descs) == len(aux_tower_descs)
        # every cell should have unique ID so we can tell where arch params are shared
        assert len(set(c.id for c in cell_descs)) == len(cell_descs)

        self._cell_descs = cell_descs
        self.aux_tower_descs = aux_tower_descs

    def clear_trainables(self)->None:
        for stem in self.model_stems:
            stem.clear_trainables()
        for attr in ['pool_op', 'logits_op']:
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

    def state_dict(self)->dict:
        return  {
                    'cell_descs': [c.state_dict() for c in self.cell_descs()],
                    'model_stems': [stem.state_dict() for stem in self.model_stems],
                    'pool_op': self.pool_op.state_dict(),
                    'logits_op': self.logits_op.state_dict()
                }

    def load_state_dict(self, state_dict)->None:
        for c, cs in utils.zip_eq(self.cell_descs(), state_dict['cell_descs']):
            c.load_state_dict(cs)
        for stem, state in utils.zip_eq(self.model_stems, state_dict['model_stems']):
            stem.load_state_dict(state)
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
            utils.write_string(filename, yaml.dump(cloned))

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
