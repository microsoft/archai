# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Evolutionary search-related classes and methods.
"""

import copy
from operator import is_
import os
import pickle
import random
import time
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import imageio
import numpy as np
import yaml
from matplotlib import pyplot as plt

from archai.common import utils
from archai.nlp.models.model_loader import load_model_from_args
from archai.nlp.nas.nas_utils.converter import Converter
from archai.nlp.nas.nas_utils.dispatcher import check_job_status, create_jobs
from archai.nlp.nas.nas_utils.pareto_front import calculate_convex_hull, find_pareto_points
from archai.nlp.nas.nas_utils.parser import (parse_results_from_amulet,
                                             parse_values_from_yaml)
from archai.nlp.nas.search_utils.constraints import (measure_inference_latency,
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
                 n_layer_choice: Optional[List[int]] = [5, 6, 7, 8],
                 d_model_choice: Optional[List[int]] = [64, 128, 256, 512],
                 d_inner_choice: Optional[List[int]] = list(range(512, 2048, 50)),
                 n_head_choice: Optional[List[int]] = [2, 4, 8],
                 param_constraint: Optional[int] = 4e6,
                 latency_scale: Optional[float] = 1.0,
                 n_threads: Optional[int] = 1,
                 latency_repeat: Optional[int] = 5,
                 latency_constraint: Optional[float] = None,
                 model_type: Optional[str] = 'mem_transformer',
                 use_quantization: Optional[bool] = False,
                 **kwargs) -> None:
        """Initializes attributes.

        Args:
            results_path: Path to the folder that will save the results.
            population_size: Size of the population.
            parent_size: Size of the parent genes.
            mutation_size: Size of the mutated genes.
            mutation_prob: Probability of mutation.
            crossover_size: Size of the crossovered genes.
            n_iter: Number of search iterations.
            n_layer_choice: Possible number of layers.
            d_model_choice: Possible model's dimensions.
            d_inner_choice: Possible inner dimensions.
            n_head_choice: Possible number of attention heads.
            param_constraint: Number of parameters contraint.
            latency_scale: How much latencies should be scaled.
            n_threads: Number of inference threads.
            latency_repeat: Number of latency measurements.
            latency_constraint: Latency constraint.
            model_type: Type of model.
            use_quantization: Whether should use quantization or not.

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

        self.converter = Converter(n_layer_choice, d_model_choice, d_inner_choice, n_head_choice)
        self.gene_choice = self.converter.get_allowed_genes()
        self.gene_len = len(self.gene_choice)

        self.param_constraint = param_constraint
        self.latency_constraint = latency_constraint
        
        self.max_val_ppl = 70
        self.latency_scale = latency_scale
        self.n_threads = n_threads  # number of threads for latency measurement
        self.latency_repeat = latency_repeat # number of runs for mean latency computation
        self.use_quantization = use_quantization
        
        self.model_type = model_type
        self.model_config_defaults = load_model_from_args(model_type, cls_type='config').default

        self.best_config = None
        self.pareto = {'population': [],
                       'scores': [],
                       'params': [],
                       'latencies': [],
                       'memories': []}

        self.all_population = []
        self.all_scores = []
        self.all_params = []
        self.all_latencies = []
        self.all_memories = []

        # TODO: change to collections.Counter
        # update counts code will be more robust
        self.counts = {}

        self.profile()

    def search(self,
               pareto_search: Optional[bool] = False,
               eps: Optional[float] = None,
               use_convex_hull: Optional[bool] = False,
               start_train: Optional[int] = 0,
               train_local: Optional[bool] = False,
               n_gpus: Optional[int] = 1,
               gpu_config: Optional[str] = 'dgx1_1gpu_fp32',
               config_file: Optional[str] = 'wt103_base.yaml',
               max_step: Optional[int] = 500,
               experiment_name: Optional[str] = 'evolution',
               scheduler: Optional[str] = 'constant',
               use_valid: Optional[bool] = True,
               **kwargs) -> Dict[str, Any]:
        """Performs the actual search.

        Args:
            pareto_search: Whether to search in the vicinity of the maximum score or not.
            eps: Epsilon value.
            use_convex_hull: Whether should calculate convex hull or not.
            start_train: Search iteration that training should start.
            train_local: Whether samples should be locally trained or not.
            n_gpus: Number of GPUs.
            gpu_config: GPU configuration.
            config_file: Configuration file.
            max_step: Maximum number of training steps.
            experiment_name: Name of the experiment.
            scheduler: Learning rate scheduler.
            use_valid: Whether validation set should be used or not.

        Returns:
            (Dict[str, Any]): Best configuration.

        """

        # if pareto_search is False, only searches in the vicinity of the maximum score seen
        print('Performing {} search'.format('full-pareto' if pareto_search else 'best sample'))

        population = self.sample_random_population(self.population_size)

        self.all_population = population
        self.update_counts(population)

        logs = {'population': [],
                'params': [],
                'latencies': [],
                'memories': [],
                'parents': [],
                'parents_scores': [],
                'best_config': [],
                'pareto': []}

        parents_score = []
        parents_params = []
        parents_latencies = []
        parents_memories = []

        for i in range(self.n_iter):
            idx = 0 if i == 0 else self.parent_size
            print(f"| Start Iteration {i}:")

            do_train = True if (i >= start_train) else False

            if do_train and i == start_train:
                idx = 0
                parents_score = []
                parents_params = []
                parents_latencies = []
                parents_memories = []

                self.all_population = population
                self.all_scores = []
                self.all_params = []
                self.all_latencies = []
                self.all_memories = []

            population_scores_unseen, population_params_unseen, population_latencies_unseen, population_memories_unseen = \
                self.calculate_score(population[idx:], 
                                    do_train, 
                                    train_local, 
                                    n_gpus, 
                                    gpu_config, 
                                    config_file, 
                                    max_step,                                    
                                    experiment_name, 
                                    scheduler, 
                                    use_valid)

            population_scores = parents_score + population_scores_unseen
            population_params = parents_params + population_params_unseen
            population_latencies = parents_latencies + population_latencies_unseen
            population_memories = parents_memories + population_memories_unseen
            assert len(population_scores) == self.population_size
            print(f"| Iteration {i}, Max score: {max(population_scores)}")

            self.all_scores += population_scores_unseen
            self.all_params += population_params_unseen
            self.all_latencies += population_latencies_unseen
            self.all_memories += population_memories_unseen
            print('all_population len:', len(self.all_population), 
            'all_scores len:', len(self.all_scores), 
            'all_params len:', len(self.all_params), 
            'all_latencies len:', len(self.all_latencies),
            'all_memories len:', len(self.all_memories))

            self.update_pareto_front(eps, allow_decrease=True, use_convex_hull=use_convex_hull)

            # select parents for the next iteration
            if pareto_search: 
                count_weights = self.calculate_weighted_count()
                selected_ind = np.random.choice(len(self.pareto['population']), size=self.parent_size, p=count_weights)

                parents_population = [self.pareto['population'][m] for m in selected_ind]
                parents_score = [self.pareto['scores'][m] for m in selected_ind]
                parents_params = [self.pareto['params'][m] for m in selected_ind]
                parents_latencies = [self.pareto['latencies'][m] for m in selected_ind]
                parents_memories = [self.pareto['memories'][m] for m in selected_ind]

            else:
                sorted_ind = np.array(population_scores).argsort()[::-1][:self.parent_size]

                self.best_config = self.converter.gene_to_config(population[sorted_ind[0]])
                self.best_param = population_params[sorted_ind[0]]
                self.best_latency = population_latencies[sorted_ind[0]]
                self.best_memory = population_memories[sorted_ind[0]]

                print(f"| Config for highest score model: {self.best_config}")
                print(f"| nParams for highest score model: {population_params[sorted_ind[0]]}")
                print(f"| Latency for highest score model: {population_latencies[sorted_ind[0]]}")
                print(f"| Memory for highest score model: {population_memories[sorted_ind[0]]}")

                parents_population = [population[m] for m in sorted_ind]
                parents_score = [population_scores[m] for m in sorted_ind]
                parents_params = [population_params[m] for m in sorted_ind]
                parents_latencies = [population_latencies[m] for m in sorted_ind]
                parents_memories = [population_memories[m] for m in sorted_ind]

            mutate_population = []

            k = 0
            while k < self.mutation_size:
                mutated_gene = self.mutation(random.choices(parents_population)[0])

                if self.check_constraints(mutated_gene):
                    mutate_population.append(mutated_gene)
                    k += 1

            crossover_population = []
            k = 0
            while k < self.crossover_size:
                crossovered_gene = self.crossover(random.sample(parents_population, 2))

                if self.check_constraints(crossovered_gene):
                    crossover_population.append(crossovered_gene)
                    k += 1

            logs['population'].append(copy.deepcopy(population))
            logs['params'].append(copy.deepcopy(population_params))
            logs['latencies'].append(copy.deepcopy(population_latencies))
            logs['memories'].append(copy.deepcopy(population_memories))
            logs['parents'].append(copy.deepcopy(parents_population))
            logs['parents_scores'].append(copy.deepcopy(parents_score))
            logs['best_config'].append(copy.deepcopy(self.best_config))
            logs['pareto'].append(copy.deepcopy(self.pareto))

            path_to_pkl = os.path.join(self.results_path, f'logs_itr{i}.pkl')
            with open(path_to_pkl, 'wb') as f:
                pickle.dump({'population': logs['population'][-1],
                             'params': logs['params'][-1],
                             'latencies': logs['latencies'][-1],
                             'memories': logs['memories'][-1],
                             'parents': logs['parents'][-1],
                             'parents_scores': logs['parents_scores'][-1],
                             'best_config': logs['best_config'][-1],
                             'pareto': logs['pareto'][-1]}, f)

            population = parents_population + mutate_population + crossover_population
            assert len(population) == self.population_size
            self.update_counts(population)

            self.all_population += mutate_population + crossover_population

            self.plot_samples(iter=i, parents={'params': parents_params, 'latencies': parents_latencies, 'memories': parents_memories}, from_training=do_train)

        path_to_pkl = os.path.join(self.results_path, 'logs.pkl')
        with open(path_to_pkl, 'wb') as f:
            pickle.dump(logs, f)

        return self.best_config

    def crossover(self, genes: List[List[Any]]) -> List[List[Any]]:
        """Performs the crossover between genes.

        Args:
            genes: List of genes.

        Returns:
            (List[List[Any]]): Crossovered genes.

        """

        crossovered_gene = []

        for i in range(self.gene_len):
            if np.random.uniform() < 0.5:
                crossovered_gene.append(genes[0][i])
            else:
                crossovered_gene.append(genes[1][i])

        return crossovered_gene

    def mutation(self, gene: List[Any]) -> List[Any]:
        """Performs mutation over a single gene.

        Args:
            gene: Gene.

        Returns:
            (List[Any]): Mutated gene.

        """

        mutated_gene = []
        d_inner_min = None
        gene_choice = self.gene_choice

        for i in range(self.gene_len):
            if i == 1:
                d_inner_min = int(1.7 * mutated_gene[-1])
                gene_choice = self.converter.get_allowed_genes(d_inner_min=d_inner_min)

            if np.random.uniform() < self.mutation_prob:
                mutated_gene.append(random.choices(gene_choice[i])[0])
            else:
                mutated_gene.append(gene[i])

        return mutated_gene

    def calculate_score(self,
                        genes: List[List[Any]],
                        do_train: Optional[bool] = False,
                        train_local: Optional[bool] = False,
                        n_gpus: Optional[int] = 8,
                        gpu_config: Optional[str] = 'dgx1_1gpu_fp32',
                        config_file: Optional[str] = 'wt103_base.yaml',
                        max_step: Optional[int] = 500,
                        experiment_name: Optional[str] = 'evolution',
                        scheduler: Optional[str] = 'constant',
                        use_valid: Optional[bool] = True,
                        start_config: Optional[int] = 0) -> Tuple[List[float], List[int], List[float]]:
        """Calculates the scoring (objective) function.

        Args:
            genes: List of genes.
            do_train: Whether samples should be trained or not.
            train_local: Whether samples should be locally trained or not.
            n_gpus: Number of GPUs.
            gpu_config: GPU configuration.
            config_file: Configuration file.
            max_step: Maximum number of training steps.
            experiment_name: Name of the experiment.
            scheduler: Learning rate scheduler.
            use_valid: Whether validation set should be used or not.
            start_config: Starting range of the configuration to be scored.

        Returns:
            (Tuple[List[float], List[int], List[float]]): List of scores, number of parameters
                and latencies.

        """

        configs = []
        for gene in genes:
            configs.append(self.converter.gene_to_config(gene))

        configs_from_jobs = None
        if do_train and not train_local:
            t0 = time.time()
            bundle_count = (self.population_size // 4) + 1  # distributes training over 4 jobs

            exp_name, bash_fname, n_configs = create_jobs(configs, start_config, bundle_count=bundle_count, max_step=max_step, n_gpus=n_gpus, gpu_config=gpu_config, target='NLX-NDv2')
            os.system(f'bash {bash_fname}')

            time.sleep(60)
            check_job_status(exp_name, n_configs, start_config)

            # download the log file from jobs to get the ppls
            path_to_results = './amlt_logs'
            os.mkdir(path_to_results)

            command = 'amlt results {} -I "*.json"  -o {}'.format(exp_name, path_to_results)
            os.system(command)

            command = 'amlt results {} -I "*.yaml"  -o {}'.format(exp_name, path_to_results)
            os.system(command)

            val_ppls, configs_from_jobs = parse_results_from_amulet(len(genes), exp_name, path_to_results, bundle_count, n_configs, start_config)
            t1 = time.time()
            train_time = t1-t0

        scores = []
        if do_train and not train_local:
            params = copy.deepcopy((val_ppls*-1).tolist())
        else:
            params = []
        latencies = []
        memories = []
        avg_time = []

        for i, config in enumerate(configs):
            model_config = copy.deepcopy(self.model_config_defaults)
            model_config.update(config)
            model = load_model_from_args(self.model_type, **model_config)

            if configs_from_jobs is not None:
                print('checking trained models match with the population')
                for k, v in config.items():
                    assert v == configs_from_jobs[i][k]

            if do_train:
                if train_local:
                    path_to_results = utils.full_path('./amlt_logs', create=True)
                    experiment_name = 'nv_xformer_xl'

                    t0 = time.time()

                    if n_gpus == 1:
                        command = 'python '
                    else:
                        command = 'python -m torch.distributed.launch --nproc_per_node="%s" ' % n_gpus

                    command += 'archai/nlp/nvidia_transformer_xl/train.py --work_dir %s --experiment_name %s --config %s --config_file wt103_base.yaml --n_layer %s --n_head %s --d_model %s --d_inner %s --d_embed %s --div_val %s --max_step %d --scheduler constant --summary_path %s' \
                                % (path_to_results, experiment_name, gpu_config, model_config['n_layer'], parse_values_from_yaml(model_config['n_head']), model_config['d_model'],
                                parse_values_from_yaml(model_config['d_inner']), model_config['d_model'], self.model_config_defaults['div_val'], max_step, path_to_results)
                    os.system(command)

                    log_file = os.path.join(os.path.join(path_to_results, experiment_name), 'summary.yaml')
                    while not os.path.exists(log_file):
                        pass
                    with open(log_file, 'r') as f:
                        summary = yaml.load(f, Loader=yaml.FullLoader)

                    t1 = time.time()
                    avg_time.append(t1-t0)

                    key = 'valid_ppl' if use_valid else 'test_ppl'
                    params.append(-summary[key])

                    model = model.to(device='cpu')
                    model.eval()

                    os.system(f'rm {log_file}')
            else:
                n_params = model.get_params()
                n_params_attention = n_params['attention']
                n_params_ff = n_params['ff']

                params.append(n_params_attention + n_params_ff)

            latency = measure_inference_latency(model,
                                                is_quantized=self.use_quantization,
                                                n_threads=self.n_threads,
                                                n_trials=self.latency_repeat)
            latencies.append(latency)

            memory = measure_peak_memory(model,
                                         is_quantized=self.use_quantization)
            memories.append(memory)

            if do_train:
                score = (params[i]*1./self.max_val_ppl) - (latency*1./self.max_latency) * self.latency_scale
                print('individual %d -> ppl: %d, latency: %.4f, score: %.4f' % (i, -params[i], latency, score))
            else:
                score = ((n_params_attention + n_params_ff)*1./self.max_n_params) - (latency*1./self.max_latency) * self.latency_scale - (memory*1./self.max_peak_memory)
                print('individual %d -> params: %d, latency: %.4f, peak memory: %.4f, score: %.4f' % (i, n_params_attention + n_params_ff, latency, memory, score))

            scores.append(score)

        if do_train and train_local:
            train_time = np.mean(avg_time)

        if do_train:
            print('average time for training samples was %.2fs' % train_time)

        # sanity check
        assert len(scores) == len(params)
        assert len(scores) == len(latencies)
        assert len(scores) == len(memories)

        return scores, params, latencies, memories

    def check_constraints(self, gene: List[Any]) -> bool:
        """Checks whether gene fulfill constraints or not.

        Args:
            gene: Gene.

        Returns:
            (bool): Whether gene has fulfilled constraints or not.

        """

        config = self.converter.gene_to_config(gene)

        for d_inner in config['d_inner']:
            if d_inner < int(1.7*config['d_model']):
                print('gene {} did not satisfy d_inner constraint: {}<1.7*{}={}'.format(gene, d_inner, config['d_model'], int(1.7*config['d_model'])))
                return False

        model_config = copy.deepcopy(self.model_config_defaults)
        model_config.update(config)
        model = load_model_from_args(self.model_type, **model_config)

        n_params = model.get_params()
        n_params_attention = n_params['attention']
        n_params_ff = n_params['ff']

        satisfy = True

        if (n_params_attention + n_params_ff) < self.param_constraint:
            print('gene {} did not satisfy nparam threshold: {}<{}'.format(gene, n_params_attention + n_params_ff, self.param_constraint))
            return False

        if self.latency_constraint is not None:
            latency = measure_inference_latency(model,
                                                is_quantized=self.use_quantization,
                                                n_threads=self.n_threads,
                                                n_trials=self.latency_repeat)
            
            if latency > self.latency_constraint:
                print('gene {} did not satisfy latency threshold: {}>{}'.format(gene, latency, self.latency_constraint))
                return False

        return satisfy

    def sample_random_population(self, sample_num: int) -> List[List[Any]]:
        """Samples a random population.

        Args:
            sample_num: Number of genes to be sampled.

        Returns:
            (List[List[Any]]): Randomly sampled population.

        """

        popu = []
        gene_choice = self.gene_choice

        i = 0
        while i < sample_num:
            samp_gene = []

            for k in range(self.gene_len):
                if k == 1:
                    d_inner_min = int(1.7 * samp_gene[-1])
                    gene_choice = self.converter.get_allowed_genes(d_inner_min=d_inner_min)

                samp_gene.append(random.choices(gene_choice[k])[0])

            if self.check_constraints(samp_gene):
                popu.append(samp_gene)
                i += 1

        return popu

    def semi_brute_force(self,
                         nsamples: int,
                         batch: Optional[int] = 1000,
                         eps: Optional[float] = None,
                         use_convex_hull: Optional[bool] = False,
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
            nsamples: Number of genes to be sampled.
            batch: Number of batched genes to conduct the brute force.
            eps: Epsilon value.
            use_convex_hull: Whether should calculate convex hull or not.
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

            population = population[:nsamples]

        else:
            population = self.sample_random_population(nsamples)

            with open(path_to_population, 'wb') as f:
                pickle.dump(population, f)

        population_scores = []

        for idx in range(0, nsamples, batch):
            curr_population = population[idx:idx+batch]
            curr_population_scores, curr_population_params, curr_population_latencies, curr_population_memories = self.calculate_score(curr_population, do_train, train_local, n_gpus, gpu_config, config_file, max_step,
                                                                                                        experiment_name, scheduler, use_valid)
            population_scores += curr_population_scores

            self.all_population += curr_population
            self.all_params += curr_population_params
            self.all_latencies += curr_population_latencies
            self.update_pareto_front(eps, allow_decrease=True, use_convex_hull=use_convex_hull)

            sorted_ind = np.array(population_scores).argsort()[::-1]

            self.best_config = self.converter.gene_to_config(self.all_population[sorted_ind[0]])
            self.best_param = self.all_params[sorted_ind[0]]
            self.best_latency = self.all_latencies[sorted_ind[0]]

            print(f"| Config for highest score model: {self.best_config}")
            print(f"| nParams for highest score model: {self.best_param}")
            print(f"| Latency for highest score model: {self.best_latency}")

            self.plot_samples(from_training=do_train)

            logs = {'population': population,
                    'params': curr_population_params,
                    'latencies': curr_population_latencies,
                    'scores': curr_population_scores,
                    'pareto': self.pareto}

            path_to_pkl = os.path.join(self.results_path, 'logs_bruteforce_{}.pkl'.format(idx))

            with open(path_to_pkl, 'wb') as f:
                print(f"=> Saving indices {idx}-{idx+batch}")
                pickle.dump(logs, f)

        sorted_ind = np.array(population_scores).argsort()[::-1]

        self.best_config = self.converter.gene_to_config(population[sorted_ind[0]])
        self.best_param = self.all_params[sorted_ind[0]]
        self.best_latency = self.all_latencies[sorted_ind[0]]

        print(f"| Config for highest score model: {self.best_config}")
        print(f"| nParams for highest score model: {self.best_param}")
        print(f"| Latency for highest score model: {self.best_latency}")

        self.plot_samples(from_training=do_train)
        self.update_pareto_front(eps, allow_decrease=True, use_convex_hull=use_convex_hull)

    def profile(self) -> None:
        """Profiles the search space.

        """

        gene = [self.gene_choice[i][-1] for i in range(self.gene_len)]
        config = self.converter.gene_to_config(gene)

        print('biggest config:', config)

        model_config = copy.deepcopy(self.model_config_defaults)
        model_config.update(config)

        biggest_model = load_model_from_args(self.model_type, **model_config)

        n_params = biggest_model.get_params()
        n_params_attention = n_params['attention']
        n_params_ff = n_params['ff']

        self.max_latency = measure_inference_latency(biggest_model, is_quantized=self.use_quantization)
        self.max_peak_memory = measure_peak_memory(biggest_model, is_quantized=self.use_quantization)
        self.max_n_params = n_params_attention + n_params_ff

        print('In this search-space -> maximum number of parameters: {}, maximum latency: {}, maximum peak memory: {}'.format(self.max_n_params, self.max_latency, self.max_peak_memory))

    def update_pareto_front(self,
                            eps: Optional[float] = None,
                            allow_decrease: Optional[bool] = True,
                            use_convex_hull: Optional[bool] = False) -> None:
        """Updates the Pareto front of the evolutionary search.

        Args:
            eps: Epsilon value.
            allow_decrease: Whether Pareto front is decreasing or not.
            use_convex_hull: Whether should calculate convex hull or not.

        """

        self.pareto = defaultdict(list)

        if use_convex_hull:
            xs = self.all_params
            ys = self.all_latencies

            hull_indices, eps_indices = calculate_convex_hull(xs, ys, eps, allow_decrease)

            all_indices = hull_indices + eps_indices

            self.pareto['population'] = [self.all_population[i] for i in all_indices]
            self.pareto['scores'] = [self.all_scores[i] for i in all_indices]
            self.pareto['params'] = [self.all_params[i] for i in all_indices]
            self.pareto['latencies'] = [self.all_latencies[i] for i in all_indices]
            self.pareto['memories'] = [self.all_memories[i] for i in all_indices]

        else:
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
            self.pareto['scores'] = [self.all_scores[i] for i in p_inds]
            self.pareto['params'] = [self.all_params[i] for i in p_inds]
            self.pareto['latencies'] = [self.all_latencies[i] for i in p_inds]
            self.pareto['memories'] = [self.all_memories[i] for i in p_inds]
            
            # earlier method but that works reasonably well on latency vs. params
            # for i in range(len(self.all_population)):
            #     this_params, this_latency = self.all_params[i], self.all_latencies[i]
            #     is_pareto = True

            #     for j in range(len(self.all_params)):
            #         params, latency = self.all_params[j], self.all_latencies[j]

            #         if (params > this_params) and (latency < this_latency):
            #             is_pareto = False
            #             break

            #     if is_pareto:
            #         self.pareto['population'].append(self.all_population[i])
            #         self.pareto['scores'].append(self.all_scores[i])
            #         self.pareto['params'].append(self.all_params[i])
            #         self.pareto['latencies'].append(self.all_latencies[i])
            #         self.pareto['memories'].append(self.all_memories[i])

        print('number of points on the pareto front:', len(self.pareto['params']))

    def update_counts(self, population: List[List[Any]]) -> None:
        """Updates the number of repeated genes.

        Args:
            population: Current population.

        """

        n_repeated = 0

        for gene in population:
            key = ','.join([str(g) for g in gene])

            if key in self.counts.keys():
                self.counts[key] += 1
                n_repeated += 1
            else:
                self.counts[key] = 1

    def calculate_weighted_count(self) -> np.array:
        """ Assigns a weight to each member of the 
        pareto frontier such that it is inversely 
        proportional to the number of times it has already
        been in the pareto frontier. This is to prevent
        the same architectures from always being in the parent pool
        
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

        # ------- scale between [0,1] to avoid numerical issues
        scaled_counts = [(count - counts_min) / counts_range for count in pareto_counts]
        count_weights = [1.0/(scaled_count + 1) for scaled_count in scaled_counts]
        count_weights = np.asarray(count_weights) / np.sum(count_weights)

        assert count_weights.size == len(self.pareto['population'])

        return count_weights

    def plot_samples(self,
                     iter: Optional[int] = None,
                     parents: Dict[str, Any] = None,
                     from_training: Optional[bool] = False) -> None:
        """Plots a set of samples from the population.

        Args:
            iter: Current iteration number.
            parents: Dictionary with parent samples.
            from_training: Whether samples have been trained or not.

        """

        if from_training:
            x_axis = np.asarray(self.all_latencies) * 1000.
            x_axis_pareto = np.asarray(self.pareto['latencies']) * 1000.

            y_axis = -np.asarray(self.all_params)
            y_axis_pareto = -np.asarray(self.pareto['params'])

            x_label = 'Latency (ms)'
            y_label = 'Val ppl'

            if self.best_config:
                x_best = self.best_latency * 1000.
                y_best = -self.best_param

            if parents:
                x_parents = np.asarray(parents['latencies']) * 1000.
                y_parents = -np.asarray(parents['params'])
        else:
            x_axis = np.asarray(self.all_params)
            x_axis_pareto = np.asarray(self.pareto['params'])

            y_axis = np.asarray(self.all_latencies) * 1000.
            y_axis_pareto = np.asarray(self.pareto['latencies']) * 1000.

            x_label = 'Decoder nParams'
            y_label = 'Latency (ms)'

            if self.best_config:
                x_best = self.best_param
                y_best = self.best_latency * 1000.

            if parents:
                x_parents = parents['params']
                y_parents = np.asarray(parents['latencies']) * 1000.

        plt.figure()
        plt.scatter(x_axis, y_axis, s=10)
        plt.scatter(x_axis_pareto, y_axis_pareto, s=10)

        if self.best_config:
            plt.scatter(x_best, y_best, c='y', s=50, marker='*', edgecolors='k', alpha=0.3)

        if parents:
            plt.scatter(x_parents, y_parents, s=5, color='tab:green')

        plt.ylabel(y_label)
        plt.xlabel(x_label)

        plt.title('Pareto Curve')
        plt.grid(axis='y')

        fname = 'pareto_latency_iter{}.png'.format(iter) if iter is not None else 'pareto_latency_bruteforce.png'
        plt.savefig(os.path.join(self.results_path, fname), bbox_inches="tight")


def run_search(args: Dict[str, Any], brute_force: Optional[bool] = False) -> None:
    """Runs the evolutionary search.

    Args:
        args: Additional arguments.
        brute_force: Whether to employ semi brute-force to the search or not.

    """

    alg = Evolution(**args)

    if brute_force:
        alg.semi_brute_force(**args)

    else:
        best_config = alg.search(**args)
        print(best_config)

        images = []

        for i in range(args['n_iter']):
            fname = os.path.join(args['results_path'], 'pareto_latency_iter{}.png'.format(i))
            images.append(imageio.imread(fname))

        imageio.mimsave(os.path.join(args['results_path'], 'search_animation.gif'), images)
