# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Evolutionary search-related classes and methods.
"""

import copy
import os
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter

import numpy as np
import plotly.graph_objects as go

from archai.nlp.models.model_loader import load_model_from_args
from archai.nlp.nas.nas_utils.converter import Converter
from archai.nlp.nas.nas_utils.dispatcher import prepare_pareto_jobs, prepare_ground_truth_jobs
from archai.nlp.nas.nas_utils.pareto_front import find_pareto_points
from archai.nlp.nas.nas_utils.constraints import (measure_inference_latency, measure_parameters,
                                                  measure_peak_memory)


class Evolution:
    """Implements the evolutionary search (Genetic Algorithm).

    """

    def __init__(self,
                 results_path: str,
                 population_size: Optional[int] = 125,
                 parent_size: Optional[int] = 25,
                 mutation_size: Optional[int] = 50,
                 mutation_prob: Optional[float] = 0.3,
                 crossover_size: Optional[int] = 50,
                 n_iter: Optional[int] = 30,
                 param_constraint_lower: Optional[int] = 5e6,
                 param_constraint_upper: Optional[int] = 12e6,
                 latency_scale: Optional[float] = 1.0,
                 n_threads: Optional[int] = 1,
                 latency_repeat: Optional[int] = 5,
                 latency_constraint_upper: Optional[float] = None,
                 model_type: Optional[str] = 'mem_transformer',
                 use_quantization: Optional[bool] = False,
                 **choices) -> None:
        """Initializes attributes.

        Args:
            results_path: Path to the folder that will save the results.
            population_size: Size of the population.
            parent_size: Size of the parent genes.
            mutation_size: Size of the mutated genes.
            mutation_prob: Probability of mutation.
            crossover_size: Size of the crossovered genes.
            n_iter: Number of search iterations.
            param_constraint_lower: Any candidate below this will get rejected.
            param_constraint_upper: Any candidate above this will get rejected.
            latency_scale: How much latencies should be scaled.
            n_threads: Number of inference threads.
            latency_repeat: Number of latency measurements.
            latency_constraint_upper: Any model which has higher latency is rejected.
            model_type: Type of model.
            use_quantization: Whether should use quantization or not.
            choices: Additional keyword arguments that represent hyperparameters choices.

        """

        self.results_path = results_path
        os.makedirs(self.results_path, exist_ok=True)

        self.population_size = population_size
        self.parent_size = parent_size
        self.mutation_size = mutation_size
        self.mutation_prob = mutation_prob
        self.crossover_size = crossover_size
        assert self.population_size == self.parent_size + self.mutation_size + self.crossover_size

        self.n_iter = n_iter

        self.param_constraint_lower = param_constraint_lower
        self.param_constraint_upper = param_constraint_upper
        self.latency_constraint_upper = latency_constraint_upper
        
        self.n_threads = n_threads  # number of threads for latency measurement
        self.latency_repeat = latency_repeat # number of runs for latency measurement
        self.use_quantization = use_quantization
        
        self.model_type = model_type
        self.model_config = load_model_from_args(model_type, cls_type='config')

        # Prevents non-available keys from being used during search
        # Also, overrides default search choices with inputted ones
        model_config_search = copy.deepcopy(self.model_config.search)
        model_config_search.update((k, v) for k, v in choices.items()
                                   if k in self.model_config.search.keys())

        self.converter = Converter(**model_config_search)
        self.gene_choice = self.converter.get_allowed_genes()
        self.gene_len = len(self.gene_choice)
        
        self.best_config = None
        self.pareto = {'population': [],
                       'params': [],
                       'latencies': [],
                       'memories': []}

        self.all_population = []
        self.all_params = []
        self.all_latencies = []
        self.all_memories = []

        self.counts = Counter()

        self.profile()

    def search(self, **kwargs) -> None:
        """Performs the actual search.

        """

        # sample initial population    
        population = self.sample_random_population(self.population_size)

        self.all_population = population
        self.update_counts(population)

        logs = {'population': [],
                'params': [],
                'latencies': [],
                'memories': [],
                'parents': [],
                'pareto': []}

        parents_params = []
        parents_latencies = []
        parents_memories = []

        for i in range(self.n_iter):
            idx = 0 if i == 0 else self.parent_size
            print(f'| Start Iteration {i}:')

            # calculate decoder params, latencies, memories
            population_params_unseen, \
            population_latencies_unseen, \
            population_memories_unseen = \
                self.calculate_memory_latency(population[idx:])

            population_params = parents_params + population_params_unseen
            population_latencies = parents_latencies + population_latencies_unseen
            population_memories = parents_memories + population_memories_unseen

            assert len(population_params) == self.population_size
            assert len(population_latencies) == self.population_size
            assert len(population_memories) == self.population_size
            
            self.all_params += population_params_unseen
            self.all_latencies += population_latencies_unseen
            self.all_memories += population_memories_unseen

            print('all_population len:', len(self.all_population), 
            'all_params len:', len(self.all_params), 
            'all_latencies len:', len(self.all_latencies),
            'all_memories len:', len(self.all_memories))

            self.update_pareto_front(is_decreasing=True)

            # select parents for the next iteration from the 
            # current estimate of the pareto frontier
            # give more weight to newer parents
            count_weights = self.calculate_weighted_count()
            selected_ind = np.random.choice(len(self.pareto['population']), size=self.parent_size, p=count_weights)

            parents_population = [self.pareto['population'][m] for m in selected_ind]
            parents_params = [self.pareto['params'][m] for m in selected_ind]
            parents_latencies = [self.pareto['latencies'][m] for m in selected_ind]
            parents_memories = [self.pareto['memories'][m] for m in selected_ind]
            
            # mutate random k subset of the parents
            # while ensuring the mutations fall within 
            # desired constraint limits
            mutated_population, k = [], 0
            while k < self.mutation_size:
                mutated_gene = self.mutation(random.choices(parents_population)[0])

                if self.check_constraints(mutated_gene):
                    mutated_population.append(mutated_gene)
                    k += 1

            # crossover random k subset of the parents
            # while ensuring the crossovers fall within
            # desired constraint limits
            crossovered_population, k = [], 0
            while k < self.crossover_size:
                crossovered_gene = self.crossover(random.sample(parents_population, 2))

                if self.check_constraints(crossovered_gene):
                    crossovered_population.append(crossovered_gene)
                    k += 1

            logs['population'].append(copy.deepcopy(population))
            logs['params'].append(copy.deepcopy(population_params))
            logs['latencies'].append(copy.deepcopy(population_latencies))
            logs['memories'].append(copy.deepcopy(population_memories))
            logs['parents'].append(copy.deepcopy(parents_population))
            logs['pareto'].append(copy.deepcopy(self.pareto))

            path_to_pkl = os.path.join(self.results_path, f'logs_itr{i}.pkl')
            with open(path_to_pkl, 'wb') as f:
                pickle.dump({'population': logs['population'][-1],
                             'params': logs['params'][-1],
                             'latencies': logs['latencies'][-1],
                             'memories': logs['memories'][-1],
                             'parents': logs['parents'][-1],
                             'pareto': logs['pareto'][-1]}, f)

            population = parents_population + mutated_population + crossovered_population
            assert len(population) == self.population_size

            self.update_counts(population)
            self.all_population += mutated_population + crossovered_population
            self.plot_samples(iter=i, parents={'params': parents_params, 'latencies': parents_latencies, 'memories': parents_memories})

        path_to_pkl = os.path.join(self.results_path, 'logs.pkl')
        with open(path_to_pkl, 'wb') as f:
            pickle.dump(logs, f)

        # generate a command line per pareto frontier point
        # which can be sent off to a cluster for training
        # TODO: do non-maximum suppression on the pareto frontier
        prepare_pareto_jobs(self.results_path, 
                            converter=self.converter,
                            path_to_save=os.path.join(self.results_path, "all_pareto_jobs"))    

        # generate command line for submitting all the jobs
        prepare_ground_truth_jobs(self.results_path,
                                 self.converter,
                                 max_step=40000,
                                 start_config=0,
                                 n_jobs=20,
                                 n_gpus=8,
                                 model_type=self.model_type,
                                 gpu_config='dgx1_8gpu_fp32',
                                 path_to_save=os.path.join(self.results_path, "all_visited_jobs")) 


    def crossover(self, genes: List[List[Any]]) -> List[List[Any]]:
        """Performs the crossover between genes.

        Args:
            genes: List of genes.

        Returns:
            (List[List[Any]]): Crossovered genes.

        """

        crossovered_gene = []

        for k in range(self.gene_len):
            if np.random.uniform() < 0.5:
                crossovered_gene.append(genes[0][k])
            else:
                crossovered_gene.append(genes[1][k])

        return crossovered_gene


    def mutation(self, gene: List[Any]) -> List[Any]:
        """Performs mutation over a single gene.

        Args:
            gene: Gene.

        Returns:
            (List[Any]): Mutated gene.

        """
        mutated_gene = []
        gene_choice = self.gene_choice

        for k in range(self.gene_len):
            if k == 1:
                gene_choice = self.converter.get_allowed_genes()

            if np.random.uniform() < self.mutation_prob:
                mutated_gene.append(random.choices(gene_choice[k])[0])
            else:
                mutated_gene.append(gene[k])

        return mutated_gene


    def calculate_memory_latency(self,
                        genes: List[List[Any]],
                        ) -> Tuple[List[int], List[float], List[float]]:
        """Calculates the memory and latency.

        Args:
            genes: List of genes.

        Returns:
            (Tuple[List[int], List[float], List[float]]): List of number of parameters
            latencies and memories. 

        """
        configs = []
        for gene in genes:
            configs.append(self.converter.gene_to_config(gene))

        configs_from_jobs = None
        
        params = []

        latencies = []
        memories = []

        for i, config in enumerate(configs):
            model_config = copy.deepcopy(self.model_config.default)
            model_config.update(config)
            model = load_model_from_args(self.model_type, **model_config)

            if configs_from_jobs is not None:
                print('checking trained models match with the population')
                for k, v in config.items():
                    assert v == configs_from_jobs[i][k]
            
            total_params = measure_parameters(model, ['total'])
            decoder_params = measure_parameters(model, ['attention', 'ff'])
            params.append(decoder_params)

            latency = measure_inference_latency(model,
                                                use_quantization=self.use_quantization,
                                                n_threads=self.n_threads,
                                                n_trials=self.latency_repeat)
            latencies.append(latency)

            memory = measure_peak_memory(model, use_quantization=self.use_quantization)
            memories.append(memory)
            
        # Sanity checking
        assert len(params) == len(latencies)
        assert len(params) == len(memories)
        
        return params, latencies, memories


    def check_constraints(self, gene: List[Any]) -> bool:
        """Checks whether gene fulfill constraints or not.

        Args:
            gene: Gene.

        Returns:
            (bool): Whether gene has fulfilled constraints or not.

        """

        # Converts gene to configuration
        config = self.converter.gene_to_config(gene)

        # Loads model from current configuration
        model_config = copy.deepcopy(self.model_config.default)
        model_config.update(config)
        model = load_model_from_args(self.model_type, **model_config)

        # Checks the total number of parameters constraints
        total_params = measure_parameters(model, ['total'])

        if total_params < self.param_constraint_lower:
            print(f'gene {gene} has lower parameters {total_params} than lower threshold {self.param_constraint_lower}')
            return False

        if total_params > self.param_constraint_upper:
            print(f'gene {gene} has higher parameters {total_params} than upper threshold {self.param_constraint_upper}')
            return False

        # Checks the latency constraints
        if self.latency_constraint_upper is not None:
            latency = measure_inference_latency(model,
                                                use_quantization=self.use_quantization,
                                                n_threads=self.n_threads,
                                                n_trials=self.latency_repeat)
            
            if latency > self.latency_constraint_upper:
                print(f'gene {gene} has higher latency {latency} than upper latency threshold {self.latency_constraint_upper}')
                return False

        return True


    def sample_random_population(self, n_samples: int) -> List[List[Any]]:
        """Samples a random population.

        Args:
            n_samples: Number of genes to be sampled.

        Returns:
            (List[List[Any]]): Randomly sampled population.

        """

        population = []
        gene_choice = self.gene_choice

        i = 0
        while i < n_samples:
            sampled_gene = []

            for k in range(self.gene_len):
                if k == 1:
                    gene_choice = self.converter.get_allowed_genes()

                sampled_gene.append(random.choices(gene_choice[k])[0])

            if self.check_constraints(sampled_gene):
                population.append(sampled_gene)
                i += 1

        return population

    def semi_brute_force(self,
                         n_samples: int,
                         batch: Optional[int] = 1000,
                         eps: Optional[float] = None,
                         do_train: Optional[bool] = False,
                         train_local: Optional[bool] = False,
                         n_gpus: Optional[int] = 1,
                         gpu_config: Optional[str] = 'dgx1_1gpu_fp32',
                         config_file: Optional[str] = 'wt103_base.yaml',
                         max_step: Optional[int] = 500,
                         experiment_name: Optional[str] = 'evolution',
                         scheduler: Optional[str] = 'constant',
                         use_valid: Optional[bool] = True,
                         **kwargs) -> None:
        """Performs the semi brute-force.

        Args:
            n_samples: Number of genes to be sampled.
            batch: Number of batched genes to conduct the brute force.
            eps: Epsilon value.
            do_train: Whether samples should be trained or not.
            train_local: Whether samples should be locally trained or not.
            n_gpus: Number of GPUs.
            gpu_config: GPU configuration.
            config_file: Configuration file.
            max_step: Maximum number of training steps.
            experiment_name: Name of the experiment.
            scheduler: Learning rate scheduler.
            use_valid: Whether validation set should be used or not.

        """

        path_to_population = os.path.join(self.results_path, 'init_population_bruteforce.pkl')
        
        if os.path.exists(path_to_population):
            with open(path_to_population, 'rb') as f:
                population = pickle.load(f)

            population = population[:n_samples]

        else:
            population = self.sample_random_population(n_samples)

            with open(path_to_population, 'wb') as f:
                pickle.dump(population, f)

        population_scores = []

        for idx in range(0, n_samples, batch):
            curr_population = population[idx:idx+batch]
            curr_population_params, curr_population_latencies, curr_population_memories = self.calculate_memory_latency(curr_population, do_train, train_local, n_gpus, gpu_config, config_file, max_step,
                                                                                                        experiment_name, scheduler, use_valid)
            population_scores += curr_population_scores

            self.all_population += curr_population
            self.all_params += curr_population_params
            self.all_latencies += curr_population_latencies
            self.update_pareto_front(eps, is_decreasing=True)

            sorted_ind = np.array(population_scores).argsort()[::-1]

            self.best_config = self.converter.gene_to_config(self.all_population[sorted_ind[0]])
            self.best_param = self.all_params[sorted_ind[0]]
            self.best_latency = self.all_latencies[sorted_ind[0]]

            print(f'| Config for highest score model: {self.best_config}')
            print(f'| nParams for highest score model: {self.best_param}')
            print(f'| Latency for highest score model: {self.best_latency}')

            self.plot_samples()

            logs = {'population': population,
                    'params': curr_population_params,
                    'latencies': curr_population_latencies,
                    'scores': curr_population_scores,
                    'pareto': self.pareto}

            path_to_pkl = os.path.join(self.results_path, 'logs_bruteforce_{}.pkl'.format(idx))

            with open(path_to_pkl, 'wb') as f:
                print(f'=> Saving indices {idx}-{idx+batch}')
                pickle.dump(logs, f)

        sorted_ind = np.array(population_scores).argsort()[::-1]

        self.best_config = self.converter.gene_to_config(population[sorted_ind[0]])
        self.best_param = self.all_params[sorted_ind[0]]
        self.best_latency = self.all_latencies[sorted_ind[0]]

        print(f'| Config for highest score model: {self.best_config}')
        print(f'| nParams for highest score model: {self.best_param}')
        print(f'| Latency for highest score model: {self.best_latency}')

        self.plot_samples()
        self.update_pareto_front(eps, is_decreasing=True)

    def profile(self) -> None:
        """Profiles the search space.

        """

        gene = [self.gene_choice[k][-1] for k in range(self.gene_len)]
        config = self.converter.gene_to_config(gene)

        model_config = copy.deepcopy(self.model_config.default)
        model_config.update(config)

        biggest_model = load_model_from_args(self.model_type, **model_config)

        self.max_n_params =  measure_parameters(biggest_model, ['total'])
        self.max_decoder_params = measure_parameters(biggest_model, ['attention', 'ff'])
        self.max_latency = measure_inference_latency(biggest_model, use_quantization=self.use_quantization)
        self.max_peak_memory = measure_peak_memory(biggest_model, use_quantization=self.use_quantization)
        
        print(f'''Largest model in this space has: 
                {config}
                {self.max_n_params} total params
                {self.max_decoder_params} decoder params
                {self.max_latency:.4f}s latency
                {self.max_peak_memory:.4f}MB memory''')


    def update_pareto_front(self,
                            is_decreasing: Optional[bool] = True,
                            ) -> None:
        """Updates the Pareto front of the evolutionary search.

        Args:
            is_decreasing: Whether Pareto front is decreasing or not.
        """
        self.pareto = defaultdict(list)

        # pareto over params, latency, memory
        # since params higher is better for performance
        # and memory and latency decreasing is better
        # we convert params to a decreasing quantity
        # since the pareto finding function needs all of them
        # to be either decreasing or increasing.
        xs = np.array(max(self.all_params)) - np.array(self.all_params).reshape(-1,1)
        ys = np.array(self.all_latencies).reshape(-1, 1)
        zs = np.array(self.all_memories).reshape(-1,1)
        points = np.concatenate((xs, ys, zs), axis=1)
        p_inds = find_pareto_points(points, is_decreasing=True)

        assert points.shape[0] == len(self.all_population)
        assert points.shape[0] == len(self.all_params)
        assert points.shape[0] == len(self.all_latencies)
        assert points.shape[0] == len(self.all_memories)
        
        self.pareto['population'] = [self.all_population[i] for i in p_inds]
        self.pareto['params'] = [self.all_params[i] for i in p_inds]
        self.pareto['latencies'] = [self.all_latencies[i] for i in p_inds]
        self.pareto['memories'] = [self.all_memories[i] for i in p_inds]
            
        print('number of points on the pareto front:', len(self.pareto['params']))


    def update_counts(self, population: List[List[Any]]) -> None:
        """Updates the number of repeated genes.

        Args:
            population: Current population.

        """
        for gene in population:
            key = ','.join([str(g) for g in gene])
            # important to add as a dictionary 
            # else Counter counts the characters in the string
            self.counts.update({key:1})

            
    def calculate_weighted_count(self) -> np.array:
        """Assigns a weight to each member of the  pareto frontier such that it is inversely 
            proportional to the number of times it has already been in the working set population.
            
        This is used to select parents from the pareto frontier
        to prevent the same architectures from always being in the parent pool.
        
        Returns:
            (np.array): Weighted count.

        """

        pareto_counts = []

        for gene in self.pareto['population']:
            key = ','.join([str(g) for g in gene])
            pareto_counts.append(self.counts[key])

        counts_max = max(pareto_counts)
        counts_min = min(pareto_counts)
        counts_range = counts_max if (counts_max == counts_min) else (counts_max-counts_min)

        # Scales between [0,1] to avoid numerical issues
        scaled_counts = [(count - counts_min) / counts_range for count in pareto_counts]
        count_weights = [1.0/(scaled_count + 1) for scaled_count in scaled_counts]
        count_weights = np.asarray(count_weights) / np.sum(count_weights)

        assert count_weights.size == len(self.pareto['population'])

        return count_weights


    def plot_samples(self,
                     iter: Optional[int] = None,
                     parents: Dict[str, Any] = None) -> None:
        """Plots the state of search at every iteration.

        Args:
            iter: Current iteration number.
            parents: Dictionary with parent samples.

        """

        all_decoder_params = np.asarray(self.all_params)
        all_latencies = np.asarray(self.all_latencies)
        all_memories = np.asarray(self.all_memories)

        pareto_decoder_params = np.asarray(self.pareto['params'])
        pareto_latencies = np.asarray(self.pareto['latencies'])
        pareto_memories = np.asarray(self.pareto['memories'])

        if parents:
            parents_decoder_params = np.asarray(parents['params'])
            parents_latencies = np.asarray(parents['latencies'])
            parents_memories = np.asarray(parents['memories'])

        # 2D plot #decoder params vs latencies 
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=all_decoder_params, 
                                y=all_latencies, 
                                mode='markers',
                                marker_color='blue',
                                showlegend=True,
                                name='All visited architectures'))
        fig.add_trace(go.Scatter(x=pareto_decoder_params,
                                y=pareto_latencies,
                                mode='markers',
                                marker_color='red',
                                showlegend=True,
                                name='Pareto architectures'))
        if parents:
            fig.add_trace(go.Scatter(x=parents_decoder_params,
                                    y=parents_latencies,
                                    mode='markers',
                                    marker_color='green',
                                    showlegend=True,
                                    name='Parent architectures'))
        fig.update_layout(title_text=f"Decoder params vs. Latency (s) at Iteration {iter}",
                         xaxis_title="Decoder params",
                         yaxis_title="Latency (s)")

        savename_html = os.path.join(self.results_path, f'decoder_params_vs_latency_iter_{iter}.html')
        savename_png = os.path.join(self.results_path, f'decoder_params_vs_latency_iter_{iter}.png')

        fig.write_html(savename_html)
        fig.write_image(savename_png, engine='kaleido', width=1500, height=1500, scale=1)

        # 2D plot #decoder params vs memories
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=all_decoder_params, 
                                y=all_memories, 
                                mode='markers',
                                marker_color='blue',
                                showlegend=True,
                                name='All visited architectures'))
        fig1.add_trace(go.Scatter(x=pareto_decoder_params,
                                y=pareto_memories,
                                mode='markers',
                                marker_color='red',
                                showlegend=True,
                                name='Pareto architectures'))
        if parents:
            fig1.add_trace(go.Scatter(x=parents_decoder_params,
                                    y=parents_memories,
                                    mode='markers',
                                    marker_color='green',
                                    showlegend=True,
                                    name='Parent architectures'))
        fig1.update_layout(title_text=f"Decoder params vs. Memory (MB) at Iteration {iter}",
                         xaxis_title="Decoder params",
                         yaxis_title="Memory (MB)")

        savename_html = os.path.join(self.results_path, f'decoder_params_vs_memory_iter_{iter}.html')
        savename_png = os.path.join(self.results_path, f'decoder_params_vs_memory_iter_{iter}.png')

        fig1.write_html(savename_html)
        fig1.write_image(savename_png, engine='kaleido', width=1500, height=1500, scale=1)

        # 3D plot decoder params vs. latencies vs. memories
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter3d(x=all_decoder_params, 
                                y=all_memories,
                                z=all_latencies, 
                                mode='markers',
                                marker_color='blue',
                                showlegend=True,
                                name='All visited architectures'))
        fig3.add_trace(go.Scatter3d(x=pareto_decoder_params,
                                y=pareto_memories,
                                z=pareto_latencies,
                                mode='markers',
                                marker_color='red',
                                showlegend=True,
                                name='Pareto architectures'))
        if parents:
            fig3.add_trace(go.Scatter3d(x=parents_decoder_params,
                                    y=parents_memories,
                                    z=parents_latencies,
                                    mode='markers',
                                    marker_color='green',
                                    showlegend=True,
                                    name='Parent architectures'))
        fig3.update_layout(scene = dict(
                                        xaxis_title="Decoder params",
                                        yaxis_title="Memory (MB)",
                                        zaxis_title="Latency (s)"))

        savename_html = os.path.join(self.results_path, f'decoder_params_vs_memory_vs_latency_iter_{iter}.html')
        savename_png = os.path.join(self.results_path, f'decoder_params_vs_memory_latency_iter_{iter}.png')

        fig3.write_html(savename_html)
        fig3.write_image(savename_png, engine='kaleido', width=1500, height=1500, scale=1)


def run_search(args: Dict[str, Any], brute_force: Optional[bool] = False) -> None:
    """Runs the evolutionary search.

    Args:
        args: Search-related arguments.
        brute_force: Whether to employ semi brute-force to the search or not.

    """

    alg = Evolution(**args)

    if brute_force:
        alg.semi_brute_force(**args)
    else:
        alg.search(**args)
