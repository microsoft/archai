import copy
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
                    edge.input_ids[0] = u
                    this_nbr._nodes[i].edges[j].input_ids[0] = u
                    edge_nbrs_cell_descs.append(this_nbr)
        return edge_nbrs_cell_descs


    @overrides
    def random_sample(self, 
                    conf_model_desc:Config, 
                    seed:Optional[int]=None)->ArchWithMetaData:
        ''' Uniform random sample an architecture '''
        config = get_conf()
        

        model_desc = self.random_model_desc_builder.build(conf_model_desc, seed=seed)
        model = Model(model_desc, affine=True)
        meta_data = {'archid': self.arch_counter}
        self.arch_counter += 1
        return ArchWithMetaData(model, meta_data)
        

    @overrides
    def get_neighbors(self, arch: ArchWithMetaData) -> List[ArchWithMetaData]:
        ''' Returns 136 neighbors (56x2 op and 24 edge neighbors)
        as used in Exploring the Loss Landscape in NAS, White et al., 2021 '''
        
        # we use archai Model for DARTS space which is subclassed from nn.Module
        assert isinstance(arch.arch, Model)
        central_desc = arch.arch.desc

        central_regular_cell = self._get_regular_cell(self, central_desc)
        central_reduction_cell = self._get_reduction_cell(self, central_desc)

        op_nbrs_regular_cell_descs = self._get_ops_neighbors(central_regular_cell)
        op_nbrs_reduction_cell_descs = self._get_ops_neighbors(central_reduction_cell)

        assert len(op_nbrs_reduction_cell_descs) == 56
        assert len(op_nbrs_regular_cell_descs) == 56

        # create deepcopy of central model desc and 
        # replace all the regular cells
        # with the descs of the nbrs
        op_nbrs = []
        for reg_cell in op_nbrs_regular_cell_descs:
            this_nbr = copy.deepcopy(central_desc)
            for i, cell_desc in enumerate(this_nbr._cell_descs):
                if cell_desc.cell_type is CellType.Regular:
                    this_nbr._cell_descs[i] = copy.deepcopy(reg_cell)
            op_nbrs.append(this_nbr)
            
        # same with reduction cells
        for red_cell in op_nbrs_reduction_cell_descs:
            this_nbr = copy.deepcopy(central_desc)
            for i, cell_desc in enumerate(this_nbr._cell_descs):
                if cell_desc.cell_type is CellType.Reduction:
                    this_nbr._cell_descs[i] = copy.deepcopy(red_cell)
            op_nbrs.append(this_nbr)
        
        assert len(op_nbrs) == 112

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
        edge_nbrs = []
        for reg_cell in edge_nbrs_regular_cell_descs:
            this_nbr = copy.deepcopy(central_desc)
            for i, cell_desc in enumerate(this_nbr._cell_descs):
                if cell_desc.cell_type is CellType.Regular:
                    this_nbr._cell_descs[i] = copy.deepcopy(reg_cell)
            edge_nbrs.append(this_nbr)

        # same with reduction cells
        for red_cell in edge_nbrs_reduction_cell_descs:
            this_nbr = copy.deepcopy(central_desc)
            for i, cell_desc in enumerate(this_nbr._cell_descs):
                if cell_desc.cell_type is CellType.Reduction:
                    this_nbr._cell_descs[i] = copy.deepcopy(red_cell)
            edge_nbrs.append(this_nbr)
        
        assert len(edge_nbrs) == 24

        # Now convert all the model descs to actual Models
        all_nbrs = op_nbrs + edge_nbrs
        all_models = [Model(nbr_desc, affine=True) for nbr_desc in all_nbrs]
        
        all_arch_meta = []
        for model in all_models:
            meta_data = {'archid': self.arch_counter}
            self.arch_counter += 1
            arch_meta = ArchWithMetaData(model, meta_data)
            all_arch_meta.append(arch_meta)
        return all_arch_meta

    








    