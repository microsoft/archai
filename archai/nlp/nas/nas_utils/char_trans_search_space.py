import copy
import os
import pickle
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from archai.nlp.models.model_loader import load_config, load_model_formula, load_model_from_config
from archai.nlp.nas.nas_utils.constraints.constraint_pipeline import TorchConstraintPipeline
from archai.nlp.nas.nas_utils.constraints.torch_constraints import measure_torch_inference_latency
from archai.nlp.nas.nas_utils.converter import Converter


class CharTransSearchSpace:
    """ Characterizes Transformer-based autoregressive language models """

    def __init__(self, 
                results_path: str,
                model_type: Optional[str] = 'hf_gpt2_flex',
                model_config: Optional[Dict[str, Any]] = None,
                use_quantization: Optional[bool] = False,
                constraint_pipeline_type: Optional[str] = 'torch',
                param_constraint_lower: Optional[int] = 5e6,
                param_constraint_upper: Optional[int] = 12e6,
                n_threads: Optional[int] = 1,
                latency_repeat: Optional[int] = 5,
                **choices) -> None:

        self.results_path = results_path
        self.use_quantization = use_quantization

        self.n_threads = n_threads
        self.latency_repeat = latency_repeat

        self.model_type = model_type
        self.model_config = load_config(model_type, config_type='default')

        self.model_config_search = load_config(model_type, config_type='search')

        # Overrides default configuration with inputted ones
        self.model_config.update((k, v) for k, v in model_config.items() 
                                 if k in self.model_config.keys() and v is not None)

        # Prevents non-available keys from being used during search
        # Also, overrides default search choices with inputted ones
        for k, v in choices.items():
            if k in self.model_config_search.keys() and v is not None:
                self.model_config_search[k]['value'] = v

        # Converts between genes and configurations
        self.converter = Converter(**self.model_config_search)
        self.allowed_genes = self.converter.get_allowed_genes()
        self.gene_size = len(self.allowed_genes)

        with open(os.path.join(self.results_path, 'converter.pkl'), 'wb') as f:
            pickle.dump(self.converter, f)

        # Counter for the number of genes occurences
        self.counts = Counter()

        # Creates a constraint pipeline based on input type (`torch` or `onnx`)
        self.constraint_pipeline_type = constraint_pipeline_type
        if constraint_pipeline_type == 'torch':
            self.pipeline = TorchConstraintPipeline(use_quantization=use_quantization,
                                                    n_threads=n_threads,
                                                    n_trials=latency_repeat)


    def characterize(self):

        # NOTE: hf_gpt2_flex specific for now

        # fix num_layers, n_heads, d_inner, vary d_model
        # ----------------------------------------------
        sampled_genes = []
        
        # sample a gene first
        sampled_gene = []
        for k in range(self.gene_size):
            sampled_gene.append(random.choices(self.allowed_genes[k])[0])
        
        # vary d_model which is index 1
        for d_model in self.allowed_genes[1]:
            gene_copy = copy.deepcopy(sampled_gene)
            gene_copy[1] = d_model
            sampled_genes.append(gene_copy)
            
        # measure memory, latency, total params and decoder params 
        # for the list of genes
        
        


        # fix num_layers, n_heads, d_model, vary d_inner


        # fix num_layers, d_model, d_inner, vary n_heads


        # fix d_model, d_inner, n_heads, vary num_layers
