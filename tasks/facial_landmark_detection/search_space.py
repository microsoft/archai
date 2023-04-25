"""Search space for facial landmark detection task"""

import copy
import json
import math
import random
import re
import sys
from hashlib import sha1
from typing import List

import numpy as np
import pandas as pd
import torch
from overrides.overrides import overrides
from pathlib import Path

from archai.common.common import logger
from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.search_space import (
    BayesOptSearchSpace,
    DiscreteSearchSpace,
    EvolutionarySearchSpace,
)

from model import CustomMobileNetV2

def _gen_tv_mobilenet ( arch_def, 
                        channel_multiplier=1.0, 
                        depth_multiplier=1.0,
                        num_classes=1000):
    """generate mobilenet v2 from torchvision. Adapted from timm source code"""
    assert len(arch_def) > 0, 'arch_def is empty'
    ir_setting = []
    for block_def in arch_def:
        parts = block_def[0].split("_")
        t = 1
        c = 32
        n = 1
        s = 1
        k = 3
        ds_block = False

        for part in parts:
            if part.startswith('ds'):
                t = 1
                ds_block = True
            elif part.startswith('r'):
                n = int(part[1:])
            elif part.startswith('s'):
                s = int(part[1:])
            elif part.startswith('e'):
                t = int(part[1:])
            elif part.startswith('c'):
                c = int(part[1:])
            elif part.startswith('k'): 
                k = int(part[1:])
            elif part.startswith('ir'):
                pass
            else:
                raise Exception(f'Invalid block definition part {part}')
        
        def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
            min_value = min_value or divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_v < round_limit * v:
                new_v += divisor
            return new_v

        if not ds_block:
            n = math.ceil(n * depth_multiplier)
            c = make_divisible(c * channel_multiplier)
        
        ir_setting.append([t, c, n, s, k])

    model = CustomMobileNetV2(inverted_residual_setting=ir_setting, dropout=0, num_classes=num_classes)

    return model

def create_model_from_search_results(archid, csv_file : str, num_classes) :
    csv_path = Path(csv_file)
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    row = df[df['archid'] == archid]
    cfg = json.loads(row['config'].to_list()[0])
    assert len(cfg) != 0, 'cfg is empty. Make sure the cvs file is valid'

    model = _gen_tv_mobilenet(cfg['arch_def'], 
                                channel_multiplier=cfg['channel_multiplier'], 
                                depth_multiplier=cfg['depth_multiplier'],
                                num_classes=num_classes)
    return model


class ConfigSearchModel(torch.nn.Module):
    """This is a wrapper class to allow the model to be used in the search space"""
    def __init__(self, model : ArchaiModel, archid: str, metadata : dict):
        super().__init__()
        self.model = model
        self.archid = archid
        self.metadata = metadata
    
    def forward(self, x):
        return self.model.forward(x)

class DiscreteSearchSpaceMobileNetV2(DiscreteSearchSpace):
    def __init__(self, args, num_classes=140):
        super().__init__()
        """ Default mobilenetv2 setting in more readable format
            t - exp factor, c - channels, n - number of block repeats, s - stride
            t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]"""
        #set up a few models configs with variable depth and width
        self.cfgs_orig = [
            {
                'arch_def': [
                    ['ds_r1_k3_s1_c16'],
                    ['ir_r2_k3_s2_e6_c24'],
                    ['ir_r3_k3_s2_e6_c32'],
                    ['ir_r4_k3_s2_e6_c64'],
                    ['ir_r3_k3_s1_e6_c96'],
                    ['ir_r3_k3_s2_e6_c160'],
                    ['ir_r1_k3_s1_e6_c320']
                ],
                'channel_multiplier': 1.00,
                'depth_multiplier': 1.00
            },
            {
                'arch_def': [
                    ['ds_r1_k3_s1_c16'],
                    ['ir_r2_k3_s2_e6_c24'],
                    ['ir_r3_k3_s2_e6_c32'],
                    ['ir_r4_k3_s2_e6_c64'],
                    ['ir_r3_k3_s1_e6_c96'],
                    ['ir_r3_k3_s2_e6_c160'],
                    ['ir_r1_k3_s1_e6_c320']
                ],
                'channel_multiplier': 0.75,
                'depth_multiplier': 0.75
            },
            {
                'arch_def': [
                    ['ds_r1_k3_s1_c16'],
                    ['ir_r2_k3_s2_e6_c24'],
                    ['ir_r3_k3_s2_e6_c32'],
                    ['ir_r4_k3_s2_e6_c64'],
                    ['ir_r3_k3_s1_e6_c96'],
                    ['ir_r3_k3_s2_e6_c160'],
                    ['ir_r1_k3_s1_e6_c320']
                ],
                'channel_multiplier': 0.5,
                'depth_multiplier': 0.5
            },
            {
                'arch_def': [
                    ['ds_r1_k3_s1_c16'],
                    ['ir_r2_k3_s2_e6_c24'],
                    ['ir_r3_k3_s2_e6_c32'],
                    ['ir_r4_k3_s2_e6_c64'],
                    ['ir_r3_k3_s1_e6_c96'],
                    ['ir_r3_k3_s2_e6_c160'],
                    ['ir_r1_k3_s1_e6_c320']
                ],
                'channel_multiplier': 1.25,
                'depth_multiplier': 1.25
            }
        ]

        self.config_all = {}
        self.arch_counter= 0
        self.num_classes = num_classes
        self.r_range = tuple(args.r_range)
        self.e_range = tuple(args.e_range)
        self.k_range = tuple(args.k_range)
        self.channel_mult_range = tuple(args.channel_mult_range)
        self.depth_mult_range = tuple(args.depth_mult_range)
        #make each run deterministic
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

    @overrides
    def random_sample(self)->ArchaiModel:
        ''' Uniform random sample an architecture, always start with a few seed models'''

        if (self.arch_counter >= 0 and self.arch_counter <= 3):
            cfg = copy.deepcopy(self.cfgs_orig[self.arch_counter])
            arch = self._create_uniq_arch(cfg)
        else:
            arch = None
            while (arch == None) : 
                cfg = self._rand_modify_config(self.cfgs_orig[0], len(self.e_range), len(self.r_range), len(self.k_range), 
                                                    len(self.channel_mult_range), len(self.depth_mult_range))
                arch = self._create_uniq_arch(cfg)
            assert (arch != None)

        logger.info(f"{sys._getframe(0).f_code.co_name} return archid = {arch.archid} with config = {arch.metadata}")
        
        return ArchaiModel(arch=arch, archid=arch.archid, metadata={'config': arch.metadata})

    @overrides
    def save_arch(self, model: ArchaiModel, file: str):
        with open(file, 'w') as fp:
            cfg = model.metadata['config']
            json.dump({'config': cfg}, fp)

    @overrides
    def load_arch(self, file: str):
        metadata = json.load(open(file))
        config = json.loads(metadata['config'])
        arch = ConfigSearchModel(config)
        
        return ArchaiModel(arch=arch, archid=arch.archid, metadata={'config': arch.metadata})

    @overrides
    def save_model_weights(self, model: ArchaiModel, file: str):
        state_dict = model.arch.get_state_dict()
        torch.save(state_dict, file)
    
    @overrides
    def load_model_weights(self, model: ArchaiModel, file: str):
        model.arch.load_state_dict(torch.load(file))

    def _mod_block_cfg(self, cfg, type: str, block: int, delta: int, curr_range) -> str:
        """modify the cfg of a particular block"""

        block_cfg = cfg['arch_def'][block][0]

        res = re.search (rf'_{type}(\d)_', block_cfg)
        if (res != None):
            curr = res.group(1)
            curr_idx = curr_range.index(int(curr))
            mod_range = curr_range[max(0, curr_idx - delta): min(len(curr_range), curr_idx + delta + 1)];
            modified = random.choice(mod_range)

            block_cfg = block_cfg[0:res.start()+2] + str(modified) + block_cfg[res.end()-1:]

        return block_cfg

    def _mod_multilier(self, cfg, type: str, delta: int, curr_range) -> int:
        "modify either channel or depth multiplier"

        curr = cfg[f'{type}_multiplier']
        curr_idx = curr_range.index(curr)
        mod_range = curr_range[max(0, curr_idx - delta): min(len(curr_range), curr_idx + delta + 1)]
        modified = random.choice(mod_range)
        return modified

    def _rand_modify_config (self, cfgs_orig, delta_e, delta_r, delta_k, delta_ch_mult, delta_depth_mult):
        """randomly choice a block and modify the corresponding config"""

        cfg = copy.deepcopy(cfgs_orig)

        block_e = random.choice(range(2, 7))
        block_cfg_e = self._mod_block_cfg(cfg, 'e', block_e, delta_e, self.e_range)
        cfg['arch_def'][block_e][0] = block_cfg_e
        
        block_r = random.choice(range(2, 6))
        block_cfg_r = self._mod_block_cfg(cfg, 'r', block_r, delta_r, self.r_range)
        cfg['arch_def'][block_r][0] = block_cfg_r
        
        block_k = random.choice(range(1, 7))
        block_cfg_k = self._mod_block_cfg(cfg, 'k', block_k, delta_k, self.k_range)
        cfg['arch_def'][block_k][0] = block_cfg_k

        cfg['channel_multiplier'] = self._mod_multilier(cfg, 'channel', delta_ch_mult, self.channel_mult_range)
        cfg['depth_multiplier'] = self._mod_multilier(cfg, 'depth', delta_depth_mult, self.depth_mult_range)

        return cfg

    def _create_uniq_arch(self, cfg):
        """create a unique arch from the config"""
        cfg_str = json.dumps(cfg)
        archid = sha1(cfg_str.encode('ascii')).hexdigest()[0:8]

        if cfg_str in list(self.config_all.values()):
            print(f"Creating duplicated model: {cfg_str} ")
        else :
            self.config_all[archid] = cfg_str
            self.arch_counter+= 1
            logger.info(f"adding model to search space config_all, archid = {archid}, config = {cfg_str}")

        model = _gen_tv_mobilenet(cfg['arch_def'], channel_multiplier=cfg['channel_multiplier'], depth_multiplier=cfg['depth_multiplier'], \
                num_classes=self.num_classes)
        arch = ConfigSearchModel(model, archid, cfg_str)

        return arch

class ConfigSearchSpaceExt(DiscreteSearchSpaceMobileNetV2, EvolutionarySearchSpace, BayesOptSearchSpace):
    """Search space for config search"""    
    @overrides
    def mutate(self, model_1: ArchaiModel) -> ArchaiModel:

        cfg_1 = json.loads(model_1.metadata['config'])
        
        arch = None
        while (arch == None):
            cfg = self._rand_modify_config(cfg_1, len(self.e_range), len(self.r_range), len(self.k_range), 
                                                len(self.channel_mult_range), len(self.depth_mult_range))
            arch = self._create_uniq_arch(cfg)
        assert (arch != None)
        logger.info(f"{sys._getframe(0).f_code.co_name} return archid = {arch.archid} with config = {arch.metadata}")
        
        return ArchaiModel(arch=arch, archid=arch.archid, metadata={'config' : arch.metadata})
    
    @overrides
    def crossover(self, model_list: List[ArchaiModel]) -> ArchaiModel:
        model_1, model_2 = model_list[:2]

        cfg_1 = json.loads(model_1.metadata['config'])
        cfg_2 = json.loads(model_2.metadata['config'])

        cfg = copy.deepcopy(cfg_1)

        arch = None
        while (arch == None) :
            for block in range(2, len(cfg['arch_def'])):
                cfg['arch_def'][block] = random.choice ((cfg_1['arch_def'][block], cfg_2['arch_def'][block]))

            cfg['channel_multiplier'] = random.choice((cfg_1['channel_multiplier'], cfg_2['channel_multiplier']))
            cfg['depth_multiplier'] = random.choice((cfg_1['depth_multiplier'], cfg_2['depth_multiplier']))
        
            arch = self._create_uniq_arch(cfg)
        assert (arch != None)
        logger.info(f"{sys._getframe(0).f_code.co_name} return archid = {arch.archid} with config = {arch.metadata}")
        
        return ArchaiModel(arch=arch, archid=arch.archid, metadata={'config' : arch.metadata})

    @overrides
    def encode(self, model: ArchaiModel) -> np.ndarray:
        #TBD, this is not needed for this implementation
        assert (False)

if __name__ == "__main__":
    """unit test for this module"""
    from torchinfo import summary
    img_size = 128
    def create_random_model(ss):

        arch = ss.random_sample()
        model = arch.arch

        model.to('cpu').eval()
        pred = model(torch.randn(1, 3, img_size, img_size))
        model_summary =summary(model, input_size=(1, 3, img_size, img_size), col_names=['input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'], device='cpu')

        return arch

    ss = DiscreteSearchSpaceMobileNetV2()  
    for i in range(0, 2): 
        archai_model = create_random_model(ss)
        print(archai_model.metadata['config'])