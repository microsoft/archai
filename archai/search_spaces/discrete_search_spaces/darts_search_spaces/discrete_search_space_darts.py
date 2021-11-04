import copy
import random
from typing import List, Optional
from overrides.overrides import overrides

from archai.algos.random.random_model_desc_builder import RandomModelDescBuilder
from archai.common.config import Config
from archai.nas.model import Model
from archai.nas.arch_meta import ArchWithMetaData
from archai.nas.discrete_search_space import DiscreteSearchSpace
from archai.nas.model_desc import CellDesc, ModelDesc, CellType
from archai.common.common import get_conf


class DiscreteSearchSpaceDARTS(DiscreteSearchSpace):

    PRIMITIVES = [
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',  # identity
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5'    
    ]

    def __init__(self):
        self.arch_counter = 0
        self.random_model_desc_builder = RandomModelDescBuilder()

        self.config = get_conf()
        self.conf_model_desc = self.config['nas']['search']['model_desc']
        self.drop_path_prob = self.config['nas']['search']['trainer']['drop_path_prob']


    def _get_regular_cell(self, model_desc:ModelDesc)->CellDesc:
        ''' Returns the first regular cell type encountered '''
        # model desc should have at least 1 cell
        assert len(model_desc.cell_descs()) >= 1 
        for cell_desc in model_desc.cell_descs():
            if cell_desc.cell_type is CellType.Regular:
                return cell_desc


    def _get_reduction_cell(self, model_desc:ModelDesc)->CellDesc:
        ''' Returns the first regular cell type encountered '''
        # model desc should have at least 1 cells
        assert len(model_desc.cell_descs()) >= 1
        for cell_desc in model_desc.cell_descs():
            if cell_desc.cell_type is CellType.Reduction:
                return cell_desc


    def _get_ops_neighbors(self, cell_desc:CellDesc)->List[CellDesc]:
        op_nbrs_cell_descs = []
        for j, node in enumerate(cell_desc._nodes):
            for k,edge in enumerate(node.edges):
                available = [op_name for op_name in DiscreteSearchSpaceDARTS.PRIMITIVES if op_name!=edge.op_desc.name]
                for op_name in available:
                    # change one op to be different
                    this_nbr = copy.deepcopy(cell_desc)
                    this_nbr._nodes[j].edges[k].op_desc.name = op_name
                    op_nbrs_cell_descs.append(this_nbr)
        return op_nbrs_cell_descs


    def _get_edge_neighbors(self, cell_desc:CellDesc)->List[CellDesc]:
        edge_nbrs_cell_descs = []
        for i, node in enumerate(cell_desc._nodes):
            # no choices for the first internal node
            if i == 0:
                continue
            used_ids = {edge.input_ids[0] for edge in node.edges}
            valid_ids = {k for k in range(i+2)}
            unused_valid_ids = valid_ids - used_ids
            for j, edge in enumerate(node.edges):
                for u in unused_valid_ids:
                    # change edge source to be different
                    this_nbr = copy.deepcopy(cell_desc)
                    this_nbr._nodes[i].edges[j].input_ids[0] = u
                    edge_nbrs_cell_descs.append(this_nbr)
        return edge_nbrs_cell_descs


    @overrides
    def random_sample(self)->ArchWithMetaData:
        ''' Uniform random sample an architecture '''        
        # this is okay as this is deterministic wrt top level seed
        # so if as long as top level conf seed changes every run
        # this will be different and generate different archs to initialize
        # local search 
        seed = random.randint(0, 1e10) 
        model_desc = self.random_model_desc_builder.build(self.conf_model_desc, seed=seed)
        model = Model(model_desc, droppath=self.drop_path_prob, affine=True)
        meta_data = {'archid': self.arch_counter}
        self.arch_counter += 1
        return ArchWithMetaData(model, meta_data)

    
    def _create_nbr(self, central_desc:ModelDesc, 
                    nbr_cells:List[CellDesc], 
                    cell_type:CellType)->ModelDesc:

        nbrs = []
        for nbr_cell in nbr_cells:
            this_nbr = copy.deepcopy(central_desc)
            for i, cell_desc in enumerate(this_nbr._cell_descs):
                if cell_desc.cell_type is cell_type:
                    this_nbr._cell_descs[i] = copy.deepcopy(nbr_cell)
            nbrs.append(this_nbr)
        return nbrs
                

    @overrides
    def get_neighbors(self, arch: ArchWithMetaData) -> List[ArchWithMetaData]:
        ''' Returns 136 neighbors (56x2 op and 24 edge neighbors)
        as used in Exploring the Loss Landscape in NAS, White et al., 2021 '''
        
        # we use archai Model for DARTS space which is subclassed from nn.Module
        assert isinstance(arch.arch, Model)
        central_desc = arch.arch.desc

        central_regular_cell = self._get_regular_cell(central_desc)
        central_reduction_cell = self._get_reduction_cell(central_desc)

        op_nbrs_regular_cell_descs = self._get_ops_neighbors(central_regular_cell)
        op_nbrs_reduction_cell_descs = self._get_ops_neighbors(central_reduction_cell)

        assert len(op_nbrs_reduction_cell_descs) == 48
        assert len(op_nbrs_regular_cell_descs) == 48

        # create deepcopy of central model desc and 
        # replace all the regular cells
        # with the descs of the nbrs
        reg_op_nbrs = self._create_nbr(central_desc, 
                                    op_nbrs_regular_cell_descs,
                                    cell_type=CellType.Regular)

        red_op_nbrs = self._create_nbr(central_desc, 
                                    op_nbrs_reduction_cell_descs,
                                    cell_type=CellType.Regular)
                            
        op_nbrs = reg_op_nbrs + red_op_nbrs
        assert len(op_nbrs) == 96

        # now create the edge neighbors where the 
        # only difference is in one of the input edges
        # there should be 24 of them
        edge_nbrs_regular_cell_descs = self._get_edge_neighbors(central_regular_cell)
        edge_nbrs_reduction_cell_descs = self._get_edge_neighbors(central_reduction_cell)

        assert len(edge_nbrs_reduction_cell_descs) == 12
        assert len(edge_nbrs_regular_cell_descs) == 12

        # create deepcopy of central model desc and 
        # replace all the regular cells
        # with the descs of the nbrs
        reg_edge_nbrs = self._create_nbr(central_desc, 
                                    edge_nbrs_regular_cell_descs,
                                    cell_type=CellType.Regular)

        red_edge_nbrs = self._create_nbr(central_desc, 
                                    edge_nbrs_reduction_cell_descs,
                                    cell_type=CellType.Reduction)

        edge_nbrs = reg_edge_nbrs + red_edge_nbrs                                            
        assert len(edge_nbrs) == 24

        # Now convert all the model descs to actual Models
        all_nbrs = op_nbrs + edge_nbrs
        all_models = [Model(nbr_desc, self.drop_path_prob, affine=True) for nbr_desc in all_nbrs]
        
        all_arch_meta = []
        for model in all_models:
            meta_data = {'archid': self.arch_counter}
            self.arch_counter += 1
            arch_meta = ArchWithMetaData(model, meta_data)
            all_arch_meta.append(arch_meta)
        return all_arch_meta

    








    