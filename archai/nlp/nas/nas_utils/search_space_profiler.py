# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Search space profiler-related classes and methods.
"""

import copy
import os
import pickle
import random
from collections import Counter, namedtuple
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go

from archai.nlp.models.model_loader import load_config, load_model_from_config, load_search_config
from archai.nlp.nas.nas_utils.constraints.constraint_pipeline import (ONNXConstraintPipeline,
                                                                      TorchConstraintPipeline)
from archai.nlp.nas.nas_utils.converter import Converter

# We use a named tuple just for not missing out the properties
GeneConstraints = namedtuple('GeneConstraints', ['config',
                                                 'gene',
                                                 'decoder_params',
                                                 'total_params',
                                                 'latency_s',
                                                 'memory_mb'])


class SearchSpaceProfiler:
    """Profiles the search space of language models.
    
    """

    def __init__(self, 
                 results_path: str,
                 model_type: Optional[str] = 'mem_transformer',
                 model_config: Optional[Dict[str, Any]] = None,
                 use_quantization: Optional[bool] = False,
                 constraint_pipeline_type: Optional[str] = 'torch',
                 n_threads: Optional[int] = 1,
                 latency_repeat: Optional[int] = 10,
                 **choices) -> None:
        """Initializes attributes.

        Args:
            results_path: Path to the folder that will save the results.
            model_type: Type of model.
            model_config: Model configuration to override default configuration.
            use_quantization: Whether should use quantization or not.
            constraint_pipeline_type: Type of constraint pipeline.
            n_threads: Number of inference threads.
            latency_repeat: Number of latency measurements.
            choices: Additional keyword arguments that represent hyperparameters choices.

        """

        self.results_path = results_path
        self.use_quantization = use_quantization

        # Number of threads and runs for latency measurement
        self.n_threads = n_threads
        self.latency_repeat = latency_repeat

        # Model's default and search configurations
        self.choices_index = {}
        self.model_type = model_type
        self.model_config = load_config(model_type).to_dict()
        self.model_search_config = load_search_config(model_type).to_dict()

        # Overrides default configuration with inputted ones
        self.model_config.update((k, v) for k, v in model_config.items() 
                                 if k in self.model_config.keys() and v is not None)

        # Prevents non-available keys from being used during search
        # Also, overrides default search choices with inputted ones
        for i, (k, v) in enumerate(choices.items()):
            # Saves the hyperparameters keys and their indices for further usage
            self.choices_index[k] = i

            if k in self.model_search_config.keys() and v is not None:
                self.model_search_config[k]['value'] = v

        # Converts between genes and configurations
        self.converter = Converter(**self.model_search_config)
        self.allowed_genes = self.converter.get_allowed_genes()
        self.gene_size = len(self.allowed_genes)

        with open(os.path.join(self.results_path, 'converter.pkl'), 'wb') as f:
            pickle.dump(self.converter, f)

        # Counter for the number of genes occurences
        self.counts = Counter()

        # Creates a pipeline based on input type (`torch` or `onnx`)
        self.constraint_pipeline_type = constraint_pipeline_type
        if constraint_pipeline_type == 'torch':
            self.pipeline = TorchConstraintPipeline(use_quantization=use_quantization,
                                                    use_training_proxy=True,
                                                    n_threads=n_threads,
                                                    n_trials=latency_repeat)
        elif constraint_pipeline_type == 'onnx':
            self.pipeline = ONNXConstraintPipeline(use_quantization=use_quantization,
                                                   n_trials=latency_repeat)

    def _measure_gene_constraints(self, genes: List[Any]) -> List[GeneConstraints]:
        """Measures constraints (through pipeline) for a set of genes.

        Args:
            genes: List of genes.

        Returns:
            (List[GeneConstraints]): List of tuples holding the measured constraints.

        """

        genes_constraints = []

        for i, gene in enumerate(genes):
            print(f'Gene: {i+1}/{len(genes)}')
            
            # Converts gene from configuration and updates the model's configuration
            config = self.converter.gene_to_config(gene)
            model_config = copy.deepcopy(self.model_config)
            model_config.update(config)

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
            genes_constraints.append(gene_constraints)

        return genes_constraints
    
    def _plot_constraints(self,
                          gene_constraints: List[GeneConstraints],
                          output_path: str,
                          title_text: str) -> None:
        """Plots the constraints for clearer visualization.

        Args:
            gene_constraints: List of tuples holding the measured constraints.
            output_path: Path to the output plots.
            title_text: Title of the plots.
        """

        all_configs = [gc.config for gc in gene_constraints]
        all_total_params = [gc.total_params for gc in gene_constraints] 
        all_latencies = [gc.latency_s for gc in gene_constraints]
        all_memories = [gc.memory_mb for gc in gene_constraints]

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

        html_path = f'{output_path}.html'
        fig.write_html(html_path)

        png_path = f'{output_path}.png'
        fig.write_image(png_path, engine='kaleido', width=1500, height=1500, scale=1)

    def run(self) -> None:
        """Runs the search space profiler.

        """

        def _sample_vary_and_plot_genes(key: str) -> None:
            """Samples a set of genes, varies the inputted key and plots their constraints.

            Args:
                key: Used to vary the hyperparameter.

            """

            index = self.choices_index[key]

            sampled_genes = []
            sampled_gene = [random.choices(self.allowed_genes[k])[0] for k in range(self.gene_size)]

            print(f'Variable: {key}')

            for variable in self.allowed_genes[index]:
                sampled_gene_copy = copy.deepcopy(sampled_gene)
                sampled_gene_copy[index] = variable

                sampled_genes.append(sampled_gene_copy)

            gene_constraints = self._measure_gene_constraints(sampled_genes)

            output_path = os.path.join(self.results_path, f'vary_{key}')
            title_text = f'Variable: {key}'
            self._plot_constraints(gene_constraints,
                                   output_path=output_path,
                                   title_text=title_text)

        # Samples, varies and plots gene constraints according to the variable key
        _sample_vary_and_plot_genes('n_layer')
        _sample_vary_and_plot_genes('d_model')
        _sample_vary_and_plot_genes('d_inner')
        _sample_vary_and_plot_genes('n_head')
