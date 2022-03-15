import copy
import os
import pickle
import random
from collections import Counter, defaultdict, namedtuple
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from archai.nlp.models.model_loader import load_config, load_model_formula, load_model_from_config
from archai.nlp.nas.nas_utils.constraints.constraint_pipeline import TorchConstraintPipeline
from archai.nlp.nas.nas_utils.constraints.torch_constraints import measure_torch_inference_latency
from archai.nlp.nas.nas_utils.converter import Converter


GeneConstraints = namedtuple('GeneConstraints', ['config', 'gene', 'decoder_params', 'total_params', 'latency_s', 'memory_mb'])

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


    def _measure_gene_constraints(self, genes:List[Any])->List[GeneConstraints]:
        # measure memory, latency, total params and decoder params 
        # for the list of genes
        all_gene_contraints = []
        for gene in genes:
            config = self.converter.gene_to_config(gene)

            model_config = copy.deepcopy(self.model_config)
            model_config.update(config)

            model = load_model_from_config(self.model_type, model_config)

            # Constraint pipeline with PyTorch
            if self.constraint_pipeline_type == 'torch':
                model = load_model_from_config(self.model_type, model_config)
                params, total_params, latency, memory = self.pipeline(model)
            
            # Constraint pipeline with ONNX
            elif self.constraint_pipeline_type == 'onnx':
                params, total_params, latency, memory = self.pipeline(self.model_type, model_config)

            gene_constraints = GeneConstraints(decoder_params=params, 
                                            total_params=total_params, 
                                            latency_s=latency, 
                                            memory_mb=memory, 
                                            config=config, 
                                            gene=gene)
            all_gene_contraints.append(gene_constraints)

        return all_gene_contraints


    def characterize(self):

        assert self.model_type == 'hf_gpt2_flex'

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

        # measure constraints
        all_gene_constraints = self._measure_gene_constraints(sampled_genes)
            
        # plot
        savename_html = os.path.join(self.results_path, 'vary_d_model.html')
        savename_png = os.path.join(self.results_path, 'vary_d_model.png')
        title_text = 'Varying d_model'
        self._plot_constraints(all_gene_constraints, 
                               savename_html=savename_html, 
                               savename_png=savename_png, 
                               title_text=title_text)

            
        # fix num_layers, n_heads, d_model, vary d_inner
        # ------------------------------------------------
        sampled_genes = []

        # sample a gene first
        sampled_gene = []
        for k in range(self.gene_size):
            sampled_gene.append(random.choices(self.allowed_genes[k])[0])
        
        # vary d_inner for the first layer which is index 2
        for di in self.allowed_genes[2]:
            gene_copy = copy.deepcopy(sampled_gene)
            gene_copy[2] = di
            sampled_genes.append(gene_copy)

        # measure constraints
        all_gene_constraints = self._measure_gene_constraints(sampled_genes)
            
        # plot
        savename_html = os.path.join(self.results_path, 'vary_d_inner_layer_1.html')
        savename_png = os.path.join(self.results_path, 'vary_d_inner_layer_1.png')
        title_text = 'Varying d_inner Layer 1'
        self._plot_constraints(all_gene_constraints, 
                               savename_html=savename_html, 
                               savename_png=savename_png, 
                               title_text=title_text)



        # fix num_layers, d_model, d_inner, vary n_heads
        # ------------------------------------------------
        sampled_genes = []

        # sample a gene first
        sampled_gene = []
        for k in range(self.gene_size):
            sampled_gene.append(random.choices(self.allowed_genes[k])[0])
        
        # vary n_heads which is last index
        for n_heads in self.allowed_genes[-1]:
            gene_copy = copy.deepcopy(sampled_gene)
            gene_copy[-1] = n_heads
            sampled_genes.append(gene_copy)

        # measure constraints
        all_gene_constraints = self._measure_gene_constraints(sampled_genes)
            
        # plot
        savename_html = os.path.join(self.results_path, 'vary_n_heads.html')
        savename_png = os.path.join(self.results_path, 'vary_n_heads.png')
        title_text = 'Varying n_heads'
        self._plot_constraints(all_gene_constraints, 
                               savename_html=savename_html, 
                               savename_png=savename_png, 
                               title_text=title_text)

        # fix d_model, d_inner, n_heads, vary num_layers
        # ------------------------------------------------
        sampled_genes = []
        
        # sample a gene first
        sampled_gene = []
        for k in range(self.gene_size):
            sampled_gene.append(random.choices(self.allowed_genes[k])[0])
        
        # vary num_layers which is index 0
        for nl in self.allowed_genes[0]:
            gene_copy = copy.deepcopy(sampled_gene)
            gene_copy[0] = nl
            sampled_genes.append(gene_copy)

        # measure constraints
        all_gene_constraints = self._measure_gene_constraints(sampled_genes)
            
        # plot
        savename_html = os.path.join(self.results_path, 'vary_layers.html')
        savename_png = os.path.join(self.results_path, 'vary_layers.png')
        title_text = 'Varying layers'
        self._plot_constraints(all_gene_constraints, 
                               savename_html=savename_html, 
                               savename_png=savename_png, 
                               title_text=title_text)




    def _plot_constraints(self, gene_constraints:List[GeneConstraints],
                        savename_html:str, savename_png:str, title_text:str)->None:

        all_configs = [gs.config for gs in gene_constraints]
        all_params = [gs.decoder_params for gs in gene_constraints] 
        all_total_params = [gs.total_params for gs in gene_constraints] 
        all_latencies = [gs.latency_s for gs in gene_constraints]
        all_memories = [gs.memory_mb for gs in gene_constraints]

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=all_total_params, 
                                 y=all_memories,
                                 z=all_latencies,
                                 mode='markers',
                                 marker_color='blue',
                                 showlegend=True,
                                 name='Architectures',
                                 hovertemplate='Total params: %{x:d}' + '<br>Memory (MB): %{y:.4f}<br>' + 'Latency (s): %{z:.4f}<br>' + '%{text}',
                                 text=[repr(config) for config in all_configs]))
        
        fig.update_layout(title_text=title_text,
                      scene=dict(xaxis_title='Total Params',
                                 yaxis_title='Memory (MB)',
                                 zaxis_title='Latency (s)'))

        fig.write_html(savename_html)
        fig.write_image(savename_png, engine='kaleido', width=1500, height=1500, scale=1)