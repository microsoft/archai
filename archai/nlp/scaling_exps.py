# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Extracts Pareto-frontier through Evolutionary Search, given constraints. 
"""

from imp import new_module
import numpy as np
import copy
import math

import pickle

from archai.nlp.nas.nas_utils.constraints.constraint_pipeline import TorchConstraintPipeline
from archai.nlp.models.model_loader import load_config, load_model_from_config, load_model_formula
from archai.nlp.models.hf_gpt2.config_hf_gpt2 import HfGPT2FlexConfig

def solve_quadratic(a, b, c):
    discriminant = b**2 - 4 * a * c

    if discriminant >= 0:
        x_1=(-b+math.sqrt(discriminant))/(2*a)
        x_2=(-b-math.sqrt(discriminant))/(2*a)
    else:
        return None

    # if discriminant > 0:
    #     print("The function has two distinct real roots: ", x_1, " and ", x_2)
    # elif discriminant == 0:
    #     print("The function has one double root: ", x_1)
    # else:
    #     print("The function has two complex (conjugate) roots: ", x_1, " and ", x_2)
    
    return x_1


if __name__ == '__main__':
    model_type = 'hf_gpt2_flex' #'mem_transformer'
    model_config = {
        'n_token': 267735 if model_type=='mem_transformer' else 50257,
        'n_layer': 6,
        'n_head': 8,
        'd_model': 1024,
        'd_embed': 1024,
        'd_inner': 2048,
        'div_val': 4,
        'tgt_len': 192
    }
    model = load_model_from_config(model_type, model_config)
    assert model.model.transformer.h[0].mlp.c_proj.weight.size(1) == model_config['d_model']
    orig_dec_params = model.get_params()['non_embedding'] #load_model_formula(model_type)(orig_config)['non_embedding']
    params_formula = load_model_formula(model_type)(model_config)['non_embedding']
    print('original parameter count:', orig_dec_params, params_formula)

    ## formula is:
    ###### trans_xl: n_layer * (5*d_model**2 + 7*d_model + 2*d_model*d_inner + d_inner)
    ###### gpt2: n_layer * (4*d_model**2 + 11*d_model + 2*d_model*d_inner + d_inner)

    # load model and measure parameters
    '''
    model_config = HfGPT2FlexConfig(n_layer=6, d_model=1024, hidden_size=1024, n_head=8, d_inner=2048)
    model_config.vocab_size = 267735 if model_type=='mem_transformer' else 50257
    pipeline = TorchConstraintPipeline()
    orig_config = copy.deepcopy(model_config).to_dict()
    model = load_model_from_config(model_type, orig_config)
    orig_dec_params = model.get_params()['non_embedding'] #load_model_formula(model_type)(orig_config)['non_embedding']
    params_formula = load_model_formula(model_type)(orig_config)['non_embedding']
    '''

    orig_config = copy.deepcopy(model_config)
    new_configs = []
    for n_layer in [orig_config['n_layer']] + [51, 72, 96]:#[7, 13, 17, 25, 29, 34, 41]:
        curr_config = copy.deepcopy(orig_config)
        curr_config['n_layer'] = n_layer

        # mem_Transformer
        # new_factor = orig_dec_params/(9*n_layer)
        # new_d_model = solve_quadratic(a=1, b=1, c=-new_factor)
        
        # GPT2
        new_factor = orig_dec_params/(n_layer)
        new_d_model = solve_quadratic(a=8, b=13, c=-new_factor)

        if new_d_model is not None:
            curr_config['d_model'] = math.floor(new_d_model)
            curr_config['d_model'] += curr_config['n_head'] - (curr_config['d_model'] % curr_config['n_head'])
            assert curr_config['d_model'] % curr_config['n_head'] == 0

            curr_config['d_embed'] = curr_config['d_model']
            curr_config['d_inner'] = 2 * curr_config['d_model']

            ratio = curr_config['d_model']/curr_config['n_layer']
            if ratio < 0.1:
                break
            
            model = load_model_from_config(model_type, curr_config)
            new_dec_params = model.get_params()['non_embedding']
            # new_dec_params = load_model_formula(model_type)(curr_config)['non_embedding']
            
            print('new n_layer={}, new d_model={}, parameter count:{}'.format(curr_config['n_layer'], curr_config['d_model'], new_dec_params))
            print('ratio:', ratio)
            new_configs.append(curr_config)

    with open('scale_configs.pkl', 'wb') as f:
        pickle.dump(new_configs, f)
    
