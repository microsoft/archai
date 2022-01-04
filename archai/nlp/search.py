import copy
import os
import pickle
import random
import re
import time
import types

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from numpy.core.numeric import indices

from archai.common import utils
from archai.nlp.nas.constraint_getter import get_latency, get_model, process_parameters, get_yaml_values
from archai.nlp.nas.info_getter import get_metrics, get_results

model_config_defaults = {'d_head': None,
                         'n_token': 267736,
                         'dropout': 0.1,
                         'dropatt': 0.0,
                         'd_embed': None,
                         'div_val': 4,
                         'pre_lnorm': False,
                         'tgt_len': 192,
                         'ext_len': 0,
                         'mem_len': 192,
                         'same_length': False,
                         'attn_type': 0,
                         'clamp_len': -1,
                         'sample_softmax': -1,
                         'cutoffs': [19997, 39997, 199997],
                         'tie_projs': [False, True, True, True],
                         'tie_weight': True,
                         'dtype': None,
                         'primer_conv': False,
                         'primer_sqrt': False,
                         'use_cache': False}


class Converter(object):
    def __init__(self, n_layer_choice, d_model_choice, d_inner_choice, n_head_choice, **kwargs):
        self.n_layer_choice = n_layer_choice
        self.d_model_choice = d_model_choice
        self.d_inner_choice = d_inner_choice
        self.n_head_choice = n_head_choice

        self.max_n_layer = self.n_layer_choice[-1]

    def config2gene(self, config):
        gene = []

        sample_n_layer = config['n_layer']

        gene.append(config['d_model'])
        gene.append(sample_n_layer)

        for i in range(max(self.max_n_layer, sample_n_layer)):
            if isinstance(config['d_inner'], list):
                if i < sample_n_layer:
                    gene.append(config['d_inner'][i])
                else:
                    gene.append(config['d_inner'][0])
            else:
                gene.append(config['d_inner'])

        for i in range(max(self.max_n_layer, sample_n_layer)):
            if isinstance(config['n_head'], list):
                if i < sample_n_layer:
                    gene.append(config['n_head'][i])
                else:
                    gene.append(config['n_head'][0])
            else:
                gene.append(config['n_head'])

        return gene

    def gene2config(self, gene):
        config = {'d_model': None,
                  'n_layer': None,
                  'd_inner': None,
                  'n_head': None}

        current_index = 0

        config['d_model'] = gene[current_index]
        current_index += 1

        config['n_layer'] = gene[current_index]
        current_index += 1

        config['d_inner'] = gene[current_index: current_index + config['n_layer']]
        current_index += max(self.max_n_layer, config['n_layer'])

        config['n_head'] = gene[current_index: current_index + config['n_layer']]
        current_index += max(self.max_n_layer, config['n_layer'])

        return config

    def gene2key(self, gene):
        key_list = []

        current_index = 0

        key_list += [gene[current_index]]  # d_model
        current_index += 1

        key_list += [gene[current_index]]  # n_layer
        current_index += 1

        key_list += gene[current_index: current_index + gene[1]]  # d_inner
        current_index += self.max_n_layer

        key_list += gene[current_index: current_index + gene[1]]  # n_head
        current_index += self.max_n_layer

        return ','.join(str(k) for k in key_list)

    def key2dict(self, key):
        key_list = key.split(',')
        key_list = [int(k) for k in key_list]

        model_dict = {}
        current_index = 0

        model_dict['d_model'] = key_list[current_index]  # d_model
        current_index += 1

        model_dict['n_layer'] = key_list[current_index]  # n_layer
        current_index += 1

        model_dict['d_inner'] = key_list[current_index: current_index + key_list[1]] # d_inner
        current_index += key_list[1]

        model_dict['n_head'] = key_list[current_index: current_index + key_list[1]] # n_head

        return model_dict

    def get_gene_choice(self, d_inner_min=None):
        gene_choice = []

        gene_choice.append(self.d_model_choice)
        gene_choice.append(self.n_layer_choice)

        for _ in range(self.max_n_layer):
            if d_inner_min is not None:
                gene_choice.append(list(range(d_inner_min, self.d_inner_choice[-1], 50)))
            else:
                gene_choice.append(self.d_inner_choice)

        for _ in range(self.max_n_layer):
            gene_choice.append(self.n_head_choice)

        return gene_choice


class Evolution(object):
    def __init__(self, results_path, population_size=125, parent_size=25, mutation_size=50, mutation_prob=0.3, crossover_size=50, n_iter=30,
                 n_layer_choice=[5, 6, 7, 8], d_model_choice=[64, 128, 256, 512], d_inner_choice=list(range(512, 2048, 50)), n_head_choice=[2, 4, 8],
                 param_constraint=4e6, latency_scale=1., n_threads=1, latency_repeat=5, latency_constraint=None, **kwargs):

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
        self.gene_choice = self.converter.get_gene_choice()
        self.gene_len = len(self.gene_choice)

        self.param_constraint = param_constraint
        self.latency_constraint = latency_constraint

        self.profile()
        
        self.max_val_ppl = 70
        self.latency_scale = latency_scale
        self.n_threads = n_threads  # number of threads for latency measurement
        self.latency_repeat = latency_repeat # number of runs for mean latency computation

        self.best_config = None
        self.pareto = {'population': [],
                       'scores': [],
                       'params': [],
                       'latencies': []}

        self.all_population = []
        self.all_scores = []
        self.all_params = []
        self.all_latencies = []

        self.counts = {}

    def run_evo_search(self, pareto_search=False, eps=None, use_convex_hull=False, start_train=0, train_local=False, n_gpus=1, gpu_config='dgx1_1gpu_fp32', config_file='wt103_base.yaml',
                       max_step=500, experiment_name='evolution', scheduler='constant', use_valid=True, **kwargs):
        # if pareto_search is False, only searches in the vicinity of the maximum score seen
        print('Performing {} search'.format('full-pareto' if pareto_search else 'best sample'))

        population = self.random_sample(self.population_size)

        self.all_population = population
        self.update_counts(population)

        logs = {'population': [],
                'params': [],
                'latencies': [],
                'parents': [],
                'parents_scores': [],
                'best_config': [],
                'pareto': []}

        parents_score = []
        parents_params = []
        parents_latencies = []

        for i in range(self.n_iter):
            idx = 0 if i == 0 else self.parent_size
            print(f"| Start Iteration {i}:")

            do_train = True if (i >= start_train) else False

            if do_train and i == start_train:
                idx = 0
                parents_score = []
                parents_params = []
                parents_latencies = []

                self.all_population = population
                self.all_scores = []
                self.all_params = []
                self.all_latencies = []

            population_scores_unseen, population_params_unseen, population_latencies_unseen = self.get_scores(population[idx:], do_train, train_local, n_gpus, gpu_config, config_file, max_step,
                                                                                                              experiment_name, scheduler, use_valid)
            population_scores = parents_score + population_scores_unseen
            population_params = parents_params + population_params_unseen
            population_latencies = parents_latencies + population_latencies_unseen
            assert len(population_scores) == self.population_size
            print(f"| Iteration {i}, Max score: {max(population_scores)}")

            self.all_scores += population_scores_unseen
            self.all_params += population_params_unseen
            self.all_latencies += population_latencies_unseen
            print('all_population len:', len(self.all_population), 'all_scores len:', len(self.all_scores), 'all_params len:', len(self.all_params), 'all_latencies len:', len(self.all_latencies))

            self.update_pareto_front(eps, allow_decrease=True, use_convex_hull=use_convex_hull)

            if pareto_search:
                count_weights = self.get_count_weights()
                selected_ind = np.random.choice(len(self.pareto['population']), size=self.parent_size, p=count_weights)

                parents_population = [self.pareto['population'][m] for m in selected_ind]
                parents_score = [self.pareto['scores'][m] for m in selected_ind]
                parents_params = [self.pareto['params'][m] for m in selected_ind]
                parents_latencies = [self.pareto['latencies'][m] for m in selected_ind]

            else:
                sorted_ind = np.array(population_scores).argsort()[::-1][:self.parent_size]

                self.best_config = self.converter.gene2config(population[sorted_ind[0]])
                self.best_param = population_params[sorted_ind[0]]
                self.best_latency = population_latencies[sorted_ind[0]]

                print(f"| Config for highest score model: {self.best_config}")
                print(f"| nParams for highest score model: {population_params[sorted_ind[0]]}")
                print(f"| Latency for highest score model: {population_latencies[sorted_ind[0]]}")

                parents_population = [population[m] for m in sorted_ind]
                parents_score = [population_scores[m] for m in sorted_ind]
                parents_params = [population_params[m] for m in sorted_ind]
                parents_latencies = [population_latencies[m] for m in sorted_ind]

            mutate_population = []

            k = 0
            while k < self.mutation_size:
                mutated_gene = self.mutate(random.choices(parents_population)[0])

                if self.satisfy_constraints(mutated_gene):
                    mutate_population.append(mutated_gene)
                    k += 1

            crossover_population = []
            k = 0
            while k < self.crossover_size:
                crossovered_gene = self.crossover(random.sample(parents_population, 2))

                if self.satisfy_constraints(crossovered_gene):
                    crossover_population.append(crossovered_gene)
                    k += 1

            logs['population'].append(copy.deepcopy(population))
            logs['params'].append(copy.deepcopy(population_params))
            logs['latencies'].append(copy.deepcopy(population_latencies))
            logs['parents'].append(copy.deepcopy(parents_population))
            logs['parents_scores'].append(copy.deepcopy(parents_score))
            logs['best_config'].append(copy.deepcopy(self.best_config))
            logs['pareto'].append(copy.deepcopy(self.pareto))

            path_to_pkl = os.path.join(self.results_path, f'logs_itr{i}.pkl')
            with open(path_to_pkl, 'wb') as f:
                pickle.dump({'population': logs['population'][-1],
                             'params': logs['params'][-1],
                             'latencies': logs['latencies'][-1],
                             'parents': logs['parents'][-1],
                             'parents_scores': logs['parents_scores'][-1],
                             'best_config': logs['best_config'][-1],
                             'pareto': logs['pareto'][-1]}, f)

            population = parents_population + mutate_population + crossover_population
            self.update_counts(population)

            self.all_population += mutate_population + crossover_population

            self.plot_samples(iter=i, parents={'params': parents_params, 'latencies': parents_latencies}, from_training=do_train)

        path_to_pkl = os.path.join(self.results_path, 'logs.pkl')
        with open(path_to_pkl, 'wb') as f:
            pickle.dump(logs, f)

        return self.best_config

    def crossover(self, genes):
        crossovered_gene = []

        for i in range(self.gene_len):
            if np.random.uniform() < 0.5:
                crossovered_gene.append(genes[0][i])
            else:
                crossovered_gene.append(genes[1][i])

        return crossovered_gene

    def mutate(self, gene):
        mutated_gene = []
        d_inner_min = None
        gene_choice = self.gene_choice

        for i in range(self.gene_len):
            if i == 1:
                d_inner_min = int(1.7 * mutated_gene[-1])
                gene_choice = self.converter.get_gene_choice(d_inner_min=d_inner_min)

            if np.random.uniform() < self.mutation_prob:
                mutated_gene.append(random.choices(gene_choice[i])[0])
            else:
                mutated_gene.append(gene[i])

        return mutated_gene

    def get_scores(self, genes, do_train=False, train_local=False, n_gpus=8, gpu_config='dgx1_1gpu_fp32', config_file='wt103_base.yaml',
                   max_step=500, experiment_name='evolution', scheduler='constant', use_valid=True, start_config=0):
        configs = []
        for gene in genes:
            configs.append(self.converter.gene2config(gene))

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

            command = 'amlt results {} -I "*.json"  -o {} --no-md5'.format(exp_name, path_to_results)
            os.system(command)

            command = 'amlt results {} -I "*.yaml"  -o {} --no-md5'.format(exp_name, path_to_results)
            os.system(command)

            val_ppls, configs_from_jobs = gather_amulet_results(len(genes), exp_name, path_to_results, bundle_count, n_configs, start_config)
            t1 = time.time()
            train_time = t1-t0

        scores = []
        if do_train and not train_local:
            params = copy.deepcopy((val_ppls*-1).tolist())
        else:
            params = []
        latencies = []
        avg_time = []

        for i, config in enumerate(configs):
            model_config = copy.deepcopy(model_config_defaults)
            model_config.update(config)
            model = get_model(model_config, train=(do_train and train_local))

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
                                % (path_to_results, experiment_name, gpu_config, model_config['n_layer'], get_yaml_values(model_config['n_head']), model_config['d_model'],
                                get_yaml_values(model_config['d_inner']), model_config['d_model'], model_config_defaults['div_val'], max_step, path_to_results)
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
                _, _, _, params_attention, params_ff = process_parameters(model, verbose=False)
                params.append(params_attention + params_ff)

            latency = get_latency(model, model_config, n_threads=self.n_threads, repeat=self.latency_repeat)
            latencies.append(latency)

            if do_train:
                score = (params[i]*1./self.max_val_ppl) - (latency*1./self.max_latency) * self.latency_scale
                print('indvidual %d -> ppl: %d, latency: %.4f, score: %.4f' % (i, -params[i], latency, score))
            else:
                score = ((params_attention + params_ff)*1./self.max_n_params) - (latency*1./self.max_latency) * self.latency_scale
                print('indvidual %d -> params: %d, latency: %.4f, score: %.4f' % (i, params_attention+params_ff, latency, score))

            scores.append(score)

        if do_train and train_local:
            train_time = np.mean(avg_time)

        if do_train:
            print('average time for training samples was %.2fs' % train_time)

        return scores, params, latencies

    def satisfy_constraints(self, gene):
        config = self.converter.gene2config(gene)

        for d_inner in config['d_inner']:
            if d_inner < int(1.7*config['d_model']):
                print('gene {} did not satisfy d_inner constraint: {}<1.7*{}={}'.format(gene, d_inner, config['d_model'], int(1.7*config['d_model'])))
                return False

        model_config = copy.deepcopy(model_config_defaults)
        model_config.update(config)
        model = get_model(model_config)

        _, _, _, params_attention, params_ff = process_parameters(model, verbose=False)

        satisfy = True

        if (params_attention + params_ff) < self.param_constraint:
            print('gene {} did not satisfy nparam threshold: {}<{}'.format(gene, params_attention + params_ff, self.param_constraint))
            return False

        if self.latency_constraint is not None:
            latency = get_latency(model, model_config, n_threads=self.n_threads, repeat=self.latency_repeat)
            
            if latency > self.latency_constraint:
                print('gene {} did not satisfy latency threshold: {}>{}'.format(gene, latency, self.latency_constraint))
                return False

        return satisfy

    def random_sample(self, sample_num):
        popu = []
        gene_choice = self.gene_choice

        i = 0
        while i < sample_num:
            samp_gene = []

            for k in range(self.gene_len):
                if k == 1:
                    d_inner_min = int(1.7 * samp_gene[-1])
                    gene_choice = self.converter.get_gene_choice(d_inner_min=d_inner_min)

                samp_gene.append(random.choices(gene_choice[k])[0])

            if self.satisfy_constraints(samp_gene):
                popu.append(samp_gene)
                i += 1

        return popu

    def semi_brute_force(self, nsamples, batch=1000, eps=None, use_convex_hull=False, do_train=False, train_local=False, n_gpus=1, gpu_config='dgx1_1gpu_fp32', config_file='wt103_base.yaml',
                         max_step=500, experiment_name='evolution', scheduler='constant', use_valid=True, **kwargs):
        path_to_population = os.path.join(self.results_path, 'init_population_bruteforce.pkl')
        
        if os.path.exists(path_to_population):
            with open(path_to_population, 'rb') as f:
                population = pickle.load(f)

            population = population[:nsamples]

        else:
            population = self.random_sample(nsamples)

            with open(path_to_population, 'wb') as f:
                pickle.dump(population, f)

        population_scores = []

        for idx in range(0, nsamples, batch):
            curr_population = population[idx:idx+batch]
            curr_population_scores, curr_population_params, curr_population_latencies = self.get_scores(curr_population, do_train, train_local, n_gpus, gpu_config, config_file, max_step,
                                                                                                        experiment_name, scheduler, use_valid)
            population_scores += curr_population_scores

            self.all_population += curr_population
            self.all_params += curr_population_params
            self.all_latencies += curr_population_latencies
            self.update_pareto_front(eps, allow_decrease=True, use_convex_hull=use_convex_hull)

            sorted_ind = np.array(population_scores).argsort()[::-1]

            self.best_config = self.converter.gene2config(self.all_population[sorted_ind[0]])
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

        self.best_config = self.converter.gene2config(population[sorted_ind[0]])
        self.best_param = self.all_params[sorted_ind[0]]
        self.best_latency = self.all_latencies[sorted_ind[0]]

        print(f"| Config for highest score model: {self.best_config}")
        print(f"| nParams for highest score model: {self.best_param}")
        print(f"| Latency for highest score model: {self.best_latency}")

        self.plot_samples(from_training=do_train)
        self.update_pareto_front(eps, allow_decrease=True, use_convex_hull=use_convex_hull)

    def profile(self):
        gene = [self.gene_choice[i][-1] for i in range(self.gene_len)]
        config = self.converter.gene2config(gene)

        print('biggest config:', config)

        model_config = copy.deepcopy(model_config_defaults)
        model_config.update(config)

        biggest_model = get_model(model_config)

        _, _, _, params_attention, params_ff = process_parameters(biggest_model, verbose=False)

        self.max_latency = get_latency(biggest_model, model_config)
        self.max_n_params = params_attention + params_ff

        print('In this search-space -> maximum number of parameters: {}, maximum latency: {}'.format(self.max_n_params, self.max_latency))

        return

    def update_pareto_front(self, eps=None, allow_decrease=True, use_convex_hull=False):
        self.pareto = {'population': [],
                       'scores': [],
                       'params': [],
                       'latencies': []}

        if use_convex_hull:
            xs = self.all_params
            ys = self.all_latencies

            hull_indices, eps_indices = get_convex_hull(xs, ys, eps, allow_decrease)

            all_indices = hull_indices + eps_indices

            self.pareto['population'] = [self.all_population[i] for i in all_indices]
            self.pareto['scores'] = [self.all_scores[i] for i in all_indices]
            self.pareto['params'] = [self.all_params[i] for i in all_indices]
            self.pareto['latencies'] = [self.all_latencies[i] for i in all_indices]

        else:
            for i in range(len(self.all_population)):
                this_params, this_latency = self.all_params[i], self.all_latencies[i]
                is_pareto = True

                for j in range(len(self.all_params)):
                    params, latency = self.all_params[j], self.all_latencies[j]

                    if (params > this_params) and (latency < this_latency):
                        is_pareto = False
                        break

                if is_pareto:
                    self.pareto['population'].append(self.all_population[i])
                    self.pareto['scores'].append(self.all_scores[i])
                    self.pareto['params'].append(self.all_params[i])
                    self.pareto['latencies'].append(self.all_latencies[i])

        print('number of points on the pareto front:', len(self.pareto['params']))

        return

    def update_counts(self, population):
        n_repeated = 0

        for gene in population:
            key = ','.join([str(g) for g in gene])

            if key in self.counts.keys():
                self.counts[key] += 1
                n_repeated += 1
            else:
                self.counts[key] = 1

    def get_count_weights(self):
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

        return count_weights

    def plot_samples(self, iter=None, parents=None, from_training=False):
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


def get_convex_hull(xs, ys, eps=None, allow_decrease=False, allow_increase=False, results_path=None):
    """
    Andrew's Monotone Chain Algorithm: (https://en.wikipedia.org/wiki/Graham_scan)
    Assume the data are sorted in order of xs, then the computation complexity is O(n)
    If not sorted, then a sort by x-value is applied first. The complexity becomes O(nlog(n))
    Return:
    hull_indices (list): indices for the points on the hull exactly
    eps_indices (list): indices for the points on the hull + eps tolerance
    """
    
    xs = list(xs)
    ys = list(ys)

    indices = list(range(len(xs)))
    is_monotone = True

    for i in range(1, len(xs)):
        if xs[i] < xs[i-1]:
            is_monotone = False
            break

    if not is_monotone:
        indices.sort(key=lambda i: (xs[i], ys[i]))

    def _is_on_ray_left(x1, y1, x2, y2, x3, y3, inclusive=False, epsilon=0):
        """
        Return whether x3,y3 is on the left side of the ray x1,y1 -> x2,y2.
        If inclusive, then the answer is left or on the ray.
        If otherwise, then the answer is strictly left.
        """

        val = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)

        if inclusive:
            return val >= epsilon

        return val > epsilon

    def _remove_non_hull_idx(x1, y1, idxs):
        while len(idxs) > 1:
            x2, y2 = xs[idxs[-1]], ys[idxs[-1]]
            x3, y3 = xs[idxs[-2]], ys[idxs[-2]]

            if not _is_on_ray_left(x1, y1, x2, y2, x3, y3):
                if np.abs(x1 - x2) > 1e-6 or np.abs(y1 - y2) > 1e-6:
                    # this ensures that the no points are duplicates
                    break

            del idxs[-1]

        return idxs

    hull_indices = []

    if not allow_decrease:
        xs.insert(0, xs[indices[0]]/2)
        ys.insert(0, np.min(ys).tolist())

        indices = (np.asarray(indices)+1).tolist()
        indices.insert(0, 0)

    c = 0
    min_y = float('inf')

    for idx in indices:
        x1, y1 = xs[idx], ys[idx]

        min_y = min(y1, min_y)

        hull_indices = _remove_non_hull_idx(x1, y1, hull_indices)
        hull_indices.append(idx)

        if results_path is not None:
            plt.scatter(xs, ys, label='pts', s=5)

            hull_xs = [xs[i] for i in hull_indices]
            hull_ys = [ys[i] for i in hull_indices]

            plt.scatter(hull_xs, hull_ys, c='black', label='eps-hull', s=5)
            plt.ylim((0, 1))
            plt.savefig(os.path.join(results_path, 'debug_convex_hull_{}.png'.format(c)), dpi=plt.gcf().dpi, bbox_inches='tight')
            
            c += 1

    if not allow_increase:
        # use a fake final point at (2 * x_max , y_min) to remove increasing.
        x1, y1 = xs[indices[-1]] * 2, min_y
        hull_indices = _remove_non_hull_idx(x1, y1, hull_indices)

    # compute epsilon hull (convex hull + (1+eps) band)
    eps_indices = hull_indices
    if eps is not None and eps > 0:
        eps_indices = []
        h_idx = 0  # right idx, in the hull_indices
        for idx in indices:
            x = xs[idx]
            y = ys[idx]

            if h_idx >= len(hull_indices):
                # Larger than the largest model on the hull
                y_interp = ys[hull_indices[-1]]
            elif idx == hull_indices[h_idx]:
                # critical pts on hull
                y_interp = y
                x1, y1 = x, y  # hull point to left

                h_idx += 1
                if h_idx < len(hull_indices):
                    x2, y2 = xs[hull_indices[h_idx]], ys[hull_indices[h_idx]]
                else:
                    x2, y2 = xs[indices[-1]] * 2, ys[hull_indices[-1]]
            else:
                # Between pts of hull
                try:
                    y_interp = y1 + (y2 - y1) / (x2 - x1) * (x - x1)
                    if np.isnan(y_interp):
                        y_interp = min(y1, y2)
                except:
                    # numerical issues when x2, x1 are close
                    y_interp = min(y1, y2)

            if y <= y_interp * (1. + eps):
                eps_indices.append(idx)
                assert x1 <= x and x2 >= x, "idx={} idx[h_idx-1]={} idx[h_idx]={}  x={} y={} x1={} x2={} y1={} y2={} y_interp={}".format(idx, hull_indices[h_idx-1], hull_indices[h_idx], x, y, x1, x2, y1, y2, y_interp)

    if not allow_decrease:
        hull_indices.pop(0)
        hull_indices = (np.asarray(hull_indices)-1).tolist()

        eps_indices.pop(0)
        eps_indices = (np.asarray(eps_indices)-1).tolist()

    return hull_indices, eps_indices


def test_converter():
    config = {'d_model': 512,
              'n_layer': 5,
              'd_inner': [2048, 2049, 2050, 2051, 2052],
              'n_head': [4, 6, 7, 8, 9]}

    args = {'n_layer_choice': [5, 6, 7, 8],
            'd_model_choice': [128, 256, 512],
            'd_inner_choice': list(range(512, 2049, 100)),
            'n_head_choice': [2, 4, 8]}

    converter = Converter(**args)
    gene_get = converter.config2gene(config)
    print('generated gene:', gene_get)

    config_get = converter.gene2config(gene_get)
    print('gene -> config:', config_get)

    model_config = copy.deepcopy(model_config_defaults)
    model_config.update(config_get)
    model = get_model(model_config)

    print(model)
    print('gene choices:', converter.get_gene_choice())


def test_evo_search(args, brute_force=False):
    alg = Evolution(**args)

    if brute_force:
        alg.semi_brute_force(**args)

    else:
        best_config = alg.run_evo_search(**args)
        print(best_config)

        images = []

        for i in range(args['n_iter']):
            fname = os.path.join(args['results_path'], 'pareto_latency_iter{}.png'.format(i))
            images.append(imageio.imread(fname))

        imageio.mimsave(os.path.join(args['results_path'], 'search_animation.gif'), images)


def test_convex_hull(args):
    results_path = args['results_path']

    np.random.seed(0)
    xs = np.random.uniform(size=100)
    ys = np.random.uniform(size=100) + xs + 1.0
    eps = 0

    hull_indices, indices = get_convex_hull(xs, ys, eps, allow_decrease=False, allow_increase=True)

    hull_xs = [xs[i] for i in indices]
    hull_ys = [ys[i] for i in indices]

    bound_xs = [xs[i] for i in hull_indices]
    bound_ys = [ys[i] * (1+eps) for i in hull_indices]

    plt.plot(bound_xs, bound_ys, c='red', label='eps-bound')
    plt.scatter(xs, ys, label='pts')
    plt.scatter(hull_xs, hull_ys, c='black', marker='+', label='eps-hull')
    plt.show()
    plt.savefig(os.path.join(results_path, 'debug_convex_hull.png'), dpi=plt.gcf().dpi, bbox_inches='tight')


def test_pareto(args, eps=0.01, decreasing=True):
    np.random.seed(0)
    xs = np.random.uniform(size=1000)
    scale_x = (-1) if decreasing else (1)
    ys = np.random.uniform(size=1000) + (scale_x * xs) + 1.0

    get_vanilla_pareto(args, xs, ys, eps, decreasing)


def get_vanilla_pareto(args, xs, ys, eps=0.01, decreasing=True):
    results_path = args['results_path']

    range_x = np.max(xs) - np.min(xs)

    i = 0
    pareto_indices = []
    for i in range(len(xs)):
        curr_x = xs[i]

        # find a range where the change on x axis is smaller than eps %
        indices = []
        for j in range(len(xs)):
            diff_x = np.absolute(xs[j] - curr_x)

            if diff_x <= eps*range_x:
                indices.append(j)

        # mark the pareto front point in the found range
        curr_ys = [ys[k] for k in indices]
        pareto_idx = indices[np.argmin(curr_ys)]

        if pareto_idx not in pareto_indices:
            pareto_indices.append(pareto_idx)

    print(f'initially found {len(pareto_indices)} pareto points')

    to_remove = []
    for i in pareto_indices:
        for j in pareto_indices:
            if j == i:
                continue
            if decreasing:
                if xs[i] >= xs[j] and ys[i] >= ys[j]:
                    to_remove.append(i)
                    break
            else:
                if xs[i] < xs[j] and ys[i] >= ys[j]:
                    to_remove.append(i)
                    break

    print(f'removing {len(to_remove)} non-pareto points')

    pareto_indices_pruned = [i for i in pareto_indices if i not in to_remove]

    pareto_xs = [xs[i] for i in pareto_indices_pruned]
    pareto_ys = [ys[i] for i in pareto_indices_pruned]

    plt.scatter(xs, ys, s=5)
    plt.scatter(pareto_xs, pareto_ys, label='pareto', s=5)
    plt.scatter([xs[i] for i in to_remove], [ys[i] for i in to_remove], c='black', marker='+', label='removed', s=5)
    plt.legend()
    plt.ylim((0, 1))
    plt.savefig(os.path.join(results_path, 'debug_pareto_front.png'), dpi=plt.gcf().dpi, bbox_inches='tight')

    return pareto_indices_pruned


def check_job_status(exp_name, n_configs, start_config=0):
    pass_count = 0

    while pass_count < n_configs:
        print('Waiting for 1 minute before checking job status...')
        time.sleep(60)

        os.system(f'amlt status  {exp_name} > tmp.txt')
        with open('tmp.txt', 'r') as f:
            lines = f.readlines()

        pass_count = 0
        for i in range(len(lines)):
            l = lines[i].split()

            if len(l) == 0:
                continue

            if ':config_' in l[0]:
                config_idx = int(re.search(':config_([0-9]+)', l[0]).group(1))
                print(f'checking status of job {config_idx}')

                if (config_idx < start_config) or (config_idx >= (start_config + n_configs)):
                    print('This job index is not in range')
                    continue

                if 'pass' in l:
                    pass_count += 1

                elif 'failed' in l:
                    assert False, f'experiment {exp_name}, job :config_{config_idx} failed'

        print(f'{pass_count} total amlt jobs finished so far.')

    os.system('rm tmp.txt')


def gather_amulet_results(population_size, exp_name, path_to_results, bundle_count, n_configs, start_config):
    keys = []

    for i in range(start_config, start_config + n_configs):
        for j in range(bundle_count):
            if len(keys) == population_size:
                break
            keys.append(f'config_{i}_j{j}')

    print(keys)

    def found_all_jobs(keys, results):
        for k in keys:
            if k not in results.keys():
                return False
        return True

    results = get_results(exp_name, path_to_results, filetypes='.json')

    while not found_all_jobs(keys, results):
        print(population_size)
        time.sleep(60)
        results = get_results(exp_name, path_to_results, filetypes='.json')

    configs = get_results(exp_name, path_to_results, filetypes='.yaml')

    results_this_experiment = {k: results[k] for k in keys}
    configs_from_jobs = {k: {'d_model': configs[k]['d_model'], 'n_layer': configs[k]['n_layer'],
                             'd_inner': configs[k]['d_inner'], 'n_head': configs[k]['n_head']} for k in keys}

    configs_list = []
    val_ppls = np.zeros(population_size)
    indices = []

    for k, v in results_this_experiment.items():
        config_num = int(re.search('config_([0-9]+)', k).group(1))
        job_num = int(re.search('j([0-9]+)', k).group(1))

        val_ppls[(config_num * bundle_count) + job_num] = v['valid_perplexity']

        configs_list.append(configs_from_jobs[k])
        indices.append((config_num * bundle_count) + job_num)

    configs_list_sorted = []
    for i in range(len(configs_list)):
        idx = indices.index(i)
        configs_list_sorted.append(configs_list[idx])

    return val_ppls, configs_list_sorted


def get_bundle_run_command(configs, max_step, n_gpus, gpu_config, is_pareto=None):
    command = []
    for i, curr_config in enumerate(configs):
        curr_config['d_embed'] = curr_config['d_model']
        curr_config['d_head'] = [curr_config['d_model'] // n_head for n_head in curr_config['n_head']]

        for k, v in curr_config.items():
            curr_config[k] = str(get_yaml_values(v))

        exp_name = 'j' + str(i) + ('_pareto' if (is_pareto is not None and is_pareto[i]) else '')

        command.append('python -m torch.distributed.launch --nproc_per_node="%s" archai/nlp/nvidia_transformer_xl/train.py --config %s \
                       --config_file wt103_base.yaml --n_layer %s --n_head %s --d_model %s --d_head %s \
                       --d_inner %s --d_embed %s --div_val %s --max_step %d --experiment_name %s'
                       % (str(n_gpus), gpu_config, curr_config['n_layer'], curr_config['n_head'], curr_config['d_model'], curr_config['d_head'], curr_config['d_inner'],
                       curr_config['d_embed'], model_config_defaults['div_val'], max_step, exp_name))

    return command


def create_jobs(all_population, start_config=0, bundle_count=50, max_step=500, n_gpus=8, gpu_config='dgx1_8gpu_fp32', target='NLX-NDv2',
                exp_name='midevolution_training_', is_pareto=None, path_to_save=None):
    path_to_configs = os.path.join('./archai/nlp/nvidia_transformer_xl', 'configs') if path_to_save is None else path_to_save
    os.makedirs(path_to_configs, exist_ok=True)

    # create corresponding yaml files for amulet jobs
    n_configs = len(all_population)
    c = 0
    config_idx = start_config

    while c < n_configs:
        with open('../archaiphilly/nv_train.yaml') as file:
            amlt_config = yaml.safe_load(file)

        amlt_config['environment']['setup'] = ['set -e -o xtrace', 'pip install --user tensorboard']

        if target == 'NLX-NDV2':
            amlt_config['environment']['image'] = 'mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.0.3-cudnn8-ubuntu18.04:latest'
            amlt_config['environment']['registry'] = 'mcr.microsoft.com'
        else:
            amlt_config['environment']['image'] = 'debadeepta/pytorch:1.7.0-cuda11.0-cudnn8-devel'

        del amlt_config['search']

        amlt_config['jobs'] = [{}]
        amlt_config['jobs'][0]['name'] = 'config_{}'.format(str(config_idx))
        amlt_config['jobs'][0]['sku'] = f'G{n_gpus}'
        amlt_config['jobs'][0]['command'] = ['set -e -o xtrace', 'pip install --user -e .']

        if is_pareto is not None:
            amlt_config['jobs'][0]['command'] += get_bundle_run_command(copy.deepcopy(all_population[c:c+bundle_count]), max_step, n_gpus, gpu_config, is_pareto=is_pareto[c:c+bundle_count])
        else:
            amlt_config['jobs'][0]['command'] += get_bundle_run_command(copy.deepcopy(all_population[c:c+bundle_count]), max_step, n_gpus, gpu_config)

        config_file = 'nv_train_'+str(config_idx)+'.yaml'

        f_name = os.path.join(path_to_configs, config_file)
        with open(f_name, 'w') as file:
            yaml.dump(amlt_config, file)

        c += bundle_count
        config_idx += 1

    exp_name = exp_name + str(max_step)

    bash_f_name = 'amlt_run'
    bash_file = os.path.join(path_to_configs, bash_f_name+'.sh')

    if os.path.exists(bash_file):
        os.remove(bash_file)

    for i in range(start_config, config_idx):
        with open(bash_file, 'a') as f:
            f.write('amlt run --yes archai/nlp/nvidia_transformer_xl/configs/nv_train_{}.yaml {} -t {}\n'.format(i, exp_name, target))

    return exp_name, bash_file, config_idx-start_config


def submit_gt_jobs(args, max_step=500, start_config=0, bundle_count=50, n_gpus=8, gpu_config='dgx1_8gpu_fp32', targets=['NLX-NDv2']):
    # get amlt bash files for running all jobs to get the ground-truth Pareto
    alg = Evolution(**args)

    results_path = args['results_path']
    path_to_logs = os.path.join(results_path, 'logs.pkl')

    with open(path_to_logs, 'rb') as f:
        logs = pickle.load(f)

    # get a list of pareto points
    pareto_keys = []
    pareto_pop = logs['pareto'][-1]['population']

    for gene in pareto_pop:
        key = alg.converter.gene2key(gene)  # ','.join([str(g) for g in gene])
        if not key in pareto_keys:
            pareto_keys.append(key)

    print('number of paretos:', len(pareto_keys))
    print(len(logs['population']))

    seen = {}
    all_population = []
    is_pareto = []
    is_pareto_dict = {}
    all_latencies = {}

    for iter, pop in enumerate(logs['population']):
        unseen = 0

        for idx, gene in enumerate(pop):
            key = alg.converter.gene2key(gene)

            if not key in seen.keys():
                seen[key] = 1
                model_config = alg.converter.gene2config(gene)
                all_population.append(model_config)
                unseen += 1

                if key in pareto_keys:
                    is_pareto.append(True)
                else:
                    is_pareto.append(False)

                if iter < len(logs['latencies']):
                    all_latencies[key] = logs['latencies'][iter][idx]
                    is_pareto_dict[key] = is_pareto[-1]

    print('{} total configs and {} on the pareto'.format(len(all_population), np.sum(is_pareto)))
    assert np.sum(list(is_pareto_dict.values())) == np.sum(is_pareto)

    path_to_pkl = os.path.join(results_path, 'latencies.pkl')
    with open(path_to_pkl, 'wb') as f:
        pickle.dump(all_latencies, f)

    path_to_pkl = os.path.join(results_path, 'pareto.pkl')
    with open(path_to_pkl, 'wb') as f:
        pickle.dump(is_pareto_dict, f)

    create_jobs(all_population, start_config, bundle_count, max_step, n_gpus,
                gpu_config, targets[0], exp_name='evolution_', is_pareto=is_pareto)


def submit_pareto_front_jobs(args, is_pareto_dict, max_step=500, start_config=0, bundle_count=50, n_gpus=8, gpu_config='dgx1_8gpu_fp32', targets=['NLX-NDv2']):
    # get amlt bash files for training jobs of selected Pareto points
    alg = Evolution(**args)

    results_path = args['results_path']
    path_to_logs = os.path.join(results_path, 'logs.pkl')

    with open(path_to_logs, 'rb') as f:
        logs = pickle.load(f)

    seen = {}
    pareto_population = []
    pareto_latencies = {}

    for iter, pop in enumerate(logs['population']):
        for idx, gene in enumerate(pop):
            key = alg.converter.gene2key(gene)

            if not key in seen.keys():
                seen[key] = 1
                model_config = alg.converter.gene2config(gene)
                if is_pareto_dict[key]:
                    pareto_population.append(model_config)

                if iter < len(logs['latencies']):
                    pareto_latencies[key] = logs['latencies'][iter][idx]

    print('{} total configs and {} on the pareto'.format(len(seen.keys()), np.sum(len(pareto_population))))
    assert np.sum(list(is_pareto_dict.values())) == len(pareto_population)

    path_to_pkl = os.path.join(results_path, 'pareto_latencies.pkl')
    with open(path_to_pkl, 'wb') as f:
        pickle.dump(pareto_latencies, f)

    create_jobs(pareto_population, start_config, bundle_count, max_step, n_gpus, gpu_config, targets[0],
                exp_name='evolution_', path_to_save=os.path.join('../archaiphilly/transformer_nas', 'pareto_'+args['device_name']))


def get_diff_with_pareto(gt_latencies, gt_val_ppls, is_gt_pareto, is_proxy_pareto, min_acceptable_latency_diff=0):
    sorted_idx = np.argsort(gt_latencies)

    ppl_diff = []
    on_pareto = 0

    for i, idx in enumerate(sorted_idx):
        latency, val_ppl = gt_latencies[idx], gt_val_ppls[idx]

        if not is_proxy_pareto[idx]:
            continue

        if is_gt_pareto[idx]:
            ppl_diff.append(0.)
            on_pareto += 1
        else:
            idx_fwd, idx_bkwd = None, None

            for j in sorted_idx[i+1:]:
                if is_gt_pareto[j]:
                    diff_frwd = np.absolute(latency-gt_latencies[j])
                    idx_fwd = j
                    break

            idx_range = sorted_idx[0:i][::-1]
            for j in idx_range:
                if is_gt_pareto[j]:
                    diff_bkwd = np.absolute(latency-gt_latencies[j])
                    idx_bkwd = j
                    break

            if idx_fwd is None:
                closest_idx = idx_bkwd
            elif idx_bkwd is None or diff_frwd < diff_bkwd:
                closest_idx = idx_fwd
            else:
                closest_idx = idx_bkwd

            latency_diff = np.absolute(latency-gt_latencies[closest_idx])*1000
            print('latency difference with closest pareto point: {:1f} ms'.format(latency_diff))

            if latency_diff <= min_acceptable_latency_diff:
                ppl_diff.append(np.absolute(val_ppl-gt_val_ppls[closest_idx])*100./gt_val_ppls[closest_idx])

    if min_acceptable_latency_diff == 0:
        assert len(ppl_diff) == np.sum(is_proxy_pareto)

    print(f'{on_pareto} points out of {np.sum(is_proxy_pareto)} were on the ground-truth pareto')

    return np.mean(ppl_diff)


def get_gt_pareto(args, exp_name, path_to_dir, start_config, ppl_eps=0.1, latency_eps=0.01, hybrid=False, use_convex_hull=False,
                  min_acceptable_latency_diff=0, baseline_exp=None):
    gt_results = get_results(exp_name, os.path.join(path_to_dir, exp_name), filetypes=['config.yaml', '.json'], verbose=False)
    
    print('found %d model configurations' % len(gt_results.keys()))
    print('Loading the latencies from log file')

    results_path = args['results_path']
    path_to_pkl = os.path.join(results_path, 'latencies.pkl')

    with open(path_to_pkl, 'rb') as f:
        latencies = pickle.load(f)

    print(len(latencies.keys()))

    if baseline_exp is not None:
        print('Loading the baseline')
        latencies_baseline, params_baseline, val_ppls_baseline = analyze_baseline(args, exp_name=baseline_exp, path_to_dir=path_to_dir)

    # load previous pareto
    loaded_pareto = None
    fname = 'pareto{}'.format('' if hybrid else '_params')
    fname += '_convexHull' if use_convex_hull else ''
    path_to_pkl = os.path.join(results_path, fname+'.pkl')

    if os.path.exists(path_to_pkl):
        print('Loading proxy pareto')
        with open(path_to_pkl, 'rb') as f:
            loaded_pareto = pickle.load(f)

    alg = Evolution(**args)

    gt_latencies = []
    gt_val_ppls = []
    is_pareto = []
    gt_keys = []

    for job_name, result in gt_results.items():
        gene = alg.converter.config2gene(result)
        key = alg.converter.gene2key(gene)

        if key in latencies.keys() and not key in gt_keys:
            config_number = re.search('config_([0-9]+)_', job_name).group(1)

            if int(config_number) < start_config:
                continue

            try:
                gt_val_ppls.append(result['valid_perplexity'])
                gt_latencies.append(latencies[key])

                if loaded_pareto:
                    is_pareto.append(loaded_pareto[key])
                else:
                    is_pareto.append(True if 'pareto' in job_name else False)

                gt_keys.append(key)
            except:
                pass

    is_pareto = np.asarray(is_pareto)
    print(f'found {len(gt_val_ppls)} models with {np.sum(is_pareto)} on the proxy pareto')
    
    if use_convex_hull:
        ################# pareto extraction via convex hull #################
        assert len(gt_val_ppls) == len(is_pareto)
        is_gt_pareto = np.zeros_like(is_pareto)

        xs = gt_latencies
        ys = np.asarray(gt_val_ppls)

        gt_pareto_indices, _ = get_convex_hull(xs, ys, eps=0., allow_decrease=True, allow_increase=False)

        is_gt_pareto[gt_pareto_indices] = 1.0
    else:
        # extract the actual pareto front based on val ppl and latency
        pareto_indices = get_vanilla_pareto(args, xs=gt_latencies, ys=gt_val_ppls, eps=latency_eps, decreasing=True)
        is_gt_pareto = np.zeros_like(is_pareto)

        for i in range(len(gt_val_ppls)):
            if i in pareto_indices:
                is_gt_pareto[i] = 1

    print('{} points on the groud-truth pareto'.format(len(np.nonzero(is_gt_pareto)[0])))

    TPR = len(np.intersect1d(np.nonzero(is_gt_pareto)[0], np.nonzero(is_pareto)[0]))*100./len(np.nonzero(is_gt_pareto)[0])
    TNR = len(np.intersect1d(np.nonzero(~is_gt_pareto)[0], np.nonzero(~is_pareto)[0]))*100./len(np.nonzero(~is_gt_pareto)[0])
    
    print(f'TPR={TPR}% and TNR={TNR}%')

    mean_ppl_difference = get_diff_with_pareto(gt_latencies, gt_val_ppls, is_gt_pareto, is_pareto, min_acceptable_latency_diff=min_acceptable_latency_diff)
    
    print('mean ppl difference between proxy and gt pareto: {:.1f}%'.format(mean_ppl_difference))

    plt.figure(figsize=(5, 3))
    plt.scatter(np.asarray(gt_latencies)[~is_pareto] * 1000., np.asarray(gt_val_ppls)[~is_pareto], s=5, label='Ground-truth', color='midnightblue')
    plt.scatter(np.asarray(gt_latencies)[is_pareto] * 1000., np.asarray(gt_val_ppls)[is_pareto], s=5, label='Proxy pareto', color='tab:orange')
    
    if baseline_exp:
        plt.scatter(np.asarray(latencies_baseline) * 1000., np.asarray(val_ppls_baseline), s=25, marker='*', c='red', label='Baseline')
        plt.xlim((min(np.min(gt_latencies), np.min(latencies_baseline)) * 1000-10, np.max(gt_latencies)*1000+10))
    else:
        plt.xlim(np.min(gt_latencies)*1000-10, np.max(gt_latencies)*1000+10)

    plt.xlabel('Latency (ms)')
    plt.ylabel('Val PPL')
    plt.grid(axis='y')
    plt.legend(handletextpad=0.1, borderpad=0)
    fname = 'gt_pareto_latency{}.png'.format('' if hybrid else '_params')
    plt.savefig(os.path.join(results_path, fname), bbox_inches="tight")


def compare_w_baseline(args, exp_name, path_to_dir, start_config=None, ppl_eps=0.1, latency_eps=0.01, hybrid=False, use_convex_hull=False,
                       min_acceptable_latency_diff=0, baseline_exp=None, check_pareto=True):
    gt_results = get_results(exp_name, os.path.join(path_to_dir, exp_name), filetypes=['config.yaml', '.json'], verbose=False)
    
    print('found %d model configurations' % len(gt_results.keys()))
    print('Loading the latencies from log file')

    results_path = args['results_path']
    path_to_pkl = os.path.join(results_path, 'latencies.pkl')

    if not os.path.exists(path_to_pkl):
        path_to_pkl = os.path.join(results_path, 'pareto_latencies.pkl')

    with open(path_to_pkl, 'rb') as f:
        latencies = pickle.load(f)

    assert baseline_exp is not None, 'please provide a baseline'

    print('Loading the baseline')
    latencies_baseline, params_baseline, val_ppls_baseline = analyze_baseline(args, exp_name=baseline_exp, path_to_dir=path_to_dir)

    sorted_params_baseline = np.argsort(params_baseline)
    sorted_val_ppls_baseline = np.argsort(val_ppls_baseline)

    print('################# baseline')
    common_ratio, spr_rank = get_metrics(100, sorted_val_ppls_baseline, sorted_target=sorted_params_baseline,
                                         val_ppl_list_gt=val_ppls_baseline, val_ppl_list_target=params_baseline)

    # load previous pareto
    loaded_pareto = None
    fname = 'pareto{}'.format('' if hybrid else '_params')
    fname += '_convexHull' if use_convex_hull else ''
    path_to_pkl = os.path.join(results_path, fname+'.pkl')

    if os.path.exists(path_to_pkl):
        print('Loading proxy pareto')
        with open(path_to_pkl, 'rb') as f:
            loaded_pareto = pickle.load(f)

    alg = Evolution(**args)

    gt_latencies = []
    gt_params = []
    gt_val_ppls = []
    gt_keys = []

    for job_name, result in gt_results.items():
        gene = alg.converter.config2gene(result)
        key = alg.converter.gene2key(gene)

        if key in latencies.keys() and not key in gt_keys:
            print(job_name)
            config_number = re.search('config_([0-9]+)_', job_name).group(1)

            if int(config_number) < start_config:
                continue

            try:
                if not check_pareto or ((loaded_pareto and loaded_pareto[key]) or (not loaded_pareto and 'pareto' in job_name)):
                    model_config = copy.deepcopy(model_config_defaults)
                    model_config.update(alg.converter.gene2config(gene))
                    model = get_model(model_config)

                    _, _, _, params_attention, params_ff = process_parameters(model, verbose=False)

                    gt_val_ppls.append(result['valid_perplexity'])
                    gt_latencies.append(latencies[key])
                    gt_params.append(params_attention + params_ff)
                    gt_keys.append(key)
            except:
                pass

    gt_latencies = np.asarray(gt_latencies)
    gt_val_ppls = np.asarray(gt_val_ppls)

    print(f'found {len(gt_val_ppls)} models on the proxy pareto')

    return


def get_final_pareto_front(args, eps=0.05, hybrid=False, use_convex_hull=False):
    # exract final pareto after evolutionary search has finished
    if use_convex_hull:
        print(f'extracting the pareto using the convex hull')
    else:
        print(f'extracting the pareto with eps={eps}')

    alg = Evolution(**args)

    results_path = args['results_path']
    path_to_logs = os.path.join(results_path, 'logs.pkl')

    with open(path_to_logs, 'rb') as f:
        logs = pickle.load(f)

    # get all population
    seen_keys = []
    all_population = []
    all_params = []
    all_latencies = []
    idx = 0

    for i in range(len(logs['params'])):
        pop = logs['population'][i]

        if (args['start_train'] < args['n_iter']) and (i == args['start_train']) and hybrid:
            idx = len(all_population)

        for j, gene in enumerate(pop):
            key = alg.converter.gene2key(gene)
            if not key in seen_keys:
                seen_keys.append(key)
                all_population.append(gene)
                all_latencies.append(logs['latencies'][i][j])

                if hybrid:
                    all_params.append(logs['params'][i][j])   # -val_ppl

                else:
                    if (args['start_train'] < args['n_iter']) and (i >= args['start_train']):
                        model_config = copy.deepcopy(model_config_defaults)
                        model_config.update(alg.converter.gene2config(gene))
                        model = get_model(model_config, train=False)

                        _, _, _, params_attention, params_ff = process_parameters(model, verbose=False)

                        all_params.append(params_attention + params_ff)
                    else:
                        all_params.append(logs['params'][i][j])

    pareto = {'population': [],
              'params': [],
              'latencies': [],
              'keys': []}

    if not hybrid:
        assert idx == 0

    is_pareto_dict = {}
    if use_convex_hull:
        ################# pareto extraction via convex hull #################
        if hybrid:
            xs = all_latencies[idx:]
            ys = all_params[idx:]

        else:
            xs = all_params[idx:]
            ys = all_latencies[idx:]

        pareto_indices, _ = get_convex_hull(xs, ys, eps=0., allow_decrease=False)

        for i in range(len(all_params)):
            if i < idx and hybrid:
                is_pareto_dict[seen_keys[i]] = False
            else:
                is_pareto = (i in pareto_indices)
                if is_pareto:
                    pareto['population'].append(all_population[i])
                    pareto['params'].append(all_params[i])
                    pareto['latencies'].append(all_latencies[i])

                is_pareto_dict[seen_keys[i]] = is_pareto
    else:
        ################# faster vanilla pareto extraction on sorted values #################
        pareto_indices = get_vanilla_pareto(args, xs=all_params, ys=all_latencies, eps=eps, decreasing=False)
        
        for i in range(len(all_params)):
            if i in pareto_indices:
                is_pareto_dict[seen_keys[i]] = True

                pareto['params'].append(all_params[i])
                pareto['latencies'].append(all_latencies[i])
                pareto['keys'].append(seen_keys[i])

            else:
                is_pareto_dict[seen_keys[i]] = False

    fname = 'pareto{}'.format('' if hybrid else '_params')
    fname += '_convexHull' if use_convex_hull else ''

    path_to_logs = os.path.join(results_path, fname+'.pkl')
    with open(path_to_logs, 'wb') as f:
        pickle.dump(is_pareto_dict, f)

    path_to_logs = os.path.join(results_path, fname+'_points.pkl')
    with open(path_to_logs, 'wb') as f:
        pickle.dump(pareto, f)

    print(f'found {np.sum(list(is_pareto_dict.values()))} points on the proxy pareto')

    plt.figure()

    if (args['start_train'] < args['n_iter']) and hybrid:
        x = np.asarray(all_latencies[idx:]) * 1000.
        y = np.asarray(all_params[idx:]) * (-1)

        x_pareto = np.asarray(pareto['latencies']) * 1000.
        y_pareto = np.asarray(pareto['params']) * (-1)

        x_label = 'Latency (ms)'
        y_label = 'Val ppl'

    else:
        x = np.asarray(all_params)
        y = np.asarray(all_latencies) * 1000.

        x_pareto = np.asarray(pareto['params'])
        y_pareto = np.asarray(pareto['latencies']) * 1000.

        x_label = 'Decoder nParams'
        y_label = 'Latency (ms)'

    plt.scatter(x, y, s=5)
    plt.scatter(x_pareto, y_pareto, s=5, color='tab:orange')
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.ylim(0, 1000)
    plt.title('Pareto Curve')
    plt.grid(axis='y')
    plt.savefig(os.path.join(results_path, 'final_search_pareto{}.png'.format('' if hybrid else '_params')), bbox_inches="tight")


def analyze_baseline(args, exp_name, path_to_dir):
    baseline_results = get_results(exp_name, os.path.join(path_to_dir, exp_name), filetypes=['config.yaml', '.json'], verbose=False)

    with open(os.path.join(path_to_dir, exp_name, 'latency_summary_{}.yaml'.format(args['device_name'])), 'r') as f:
        latencies = yaml.load(f)

    with open(os.path.join(path_to_dir, exp_name, 'params.pkl'), 'rb') as f:
        params = pickle.load(f)

    latencies_list = []
    params_list = []
    val_ppls = []

    for config_name, latency in latencies.items():
        latencies_list.append(latency)
        params_list.append(params[config_name])
        val_ppls.append(baseline_results[config_name]['valid_perplexity'])

    print(f'summarized {len(latencies.keys())} baseline jobs')

    plt.figure()
    plt.scatter(np.asarray(latencies_list) * 1000., val_ppls, s=5)
    plt.xlabel('Latency (ms)')
    plt.ylabel('Val PPL')
    plt.grid(axis='y')
    fname = 'baseline_pareto_latency.png'
    plt.savefig(os.path.join(args['results_path'], fname), bbox_inches="tight")

    return latencies_list, params_list, val_ppls


if __name__ == '__main__':
    seed = 1111

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    latency_constraint = {'XeonE5-2690': 0.5,
                          'corei7': 0.5,
                          'corei5': 0.6,
                          'D3_V2': 0.8}

    '''
    #---- orig config
    args = {'default_path': './evo_search','population_size':50, 'parent_size':10, 'mutation_size':20, 'mutation_prob':0.3, 'crossover_size':20, 
            'n_iter':30, 'n_layer_choice':[3,4,5,6,7,8], 'd_model_choice':[128, 256, 512], 'd_inner_choice':list(range(512, 2049, 50))+[2048], 'n_head_choice':[2,4,8],
            'param_constraint':5e6, 'latency_scale':2., 'n_threads':1, 'latency_repeat':5, 'pareto_search':True,
            #--------------- extracting pareto
            'eps':0.05, 'use_convex_hull':False,
            #--------------- brute_force
            'nsamples':20000, 'batch':1000, 'do_train':False,
            #--------------- evaluation scheme  (set start_train to bigger than n_iter to disable training for evaluation)
            'start_train':40, 'train_local':True, 'n_gpus':4, 'gpu_config':'dgx1_4gpu_fp32', 'config_file':'wt103_base.yaml', 'max_step':500, 'experiment_name':'evolution', 
            'scheduler':'constant', 'use_valid':True}
    '''

    # ---- Bigger search-space config
    args = {'default_path': './evo_search',
            'population_size': 100,
            'parent_size': 20,
            'mutation_size': 40,
            'mutation_prob': 0.3,
            'crossover_size': 40,
            'n_iter': 10,
            'n_layer_choice': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'd_model_choice': [128, 256, 512, 650, 800],
            'd_inner_choice': list(range(512, 2049, 50))+list(range(2048, 3072, 200)),
            'n_head_choice': [2, 4, 8],
            'param_constraint': 5e6,
            'latency_scale': 2.,
            'n_threads': 1,
            'latency_repeat': 5,
            'pareto_search': True,
            'device_name': 'XeonE5-2690',
            'eps': 0.05,
            'use_convex_hull': False,
            'nsamples': 20000,
            'batch': 1000,
            'do_train': False,
            'start_train': 40,
            'train_local': True,
            'n_gpus': 1,
            'gpu_config': 'dgx1_1gpu_fp32',
            'config_file': 'wt103_base.yaml',
            'max_step': 500,
            'experiment_name': 'evolution',
            'scheduler': 'constant',
            'use_valid': True}

    args['latency_constraint'] = latency_constraint[args['device_name']]
    path_to_amlt_results = './amlt_logs'

    dir_name = 'param_threshold_{}'.format(args['param_constraint']/1e6)
    
    if args['pareto_search']:
        dir_name += '_pareto'
    if args['use_convex_hull']:
        dir_name += '_convex_hull'
    if args['start_train'] < args['n_iter']:
        dir_name += '_wTrain'

    args['results_path'] = os.path.join(args['default_path'], dir_name+'_'+args['device_name'])

    # choose from {run_search, submit_gt_jobs, extract_pareto, select_pareto, comp_w_baseline, gt_pareto}
    args['phase'] = 'run_search'

    # if True, will use convex hull to extract the final paretos, otherwise, the vanilla pareto formula
    use_convex_hull = False

    # if hybrid is true, takes the pareto on mid-search training, otherwise only looks at nparams
    hybrid = False

    if args['phase'] == 'run_search':
        results_dir = utils.full_path(args['results_path'], create=True)
        with open(os.path.join(results_dir, 'search_config.yaml'), 'w') as f:
            yaml.dump(args, f)

        test_evo_search(args, brute_force=False)

    elif args['phase'] == 'submit_gt_jobs':
        # --------------- submit ground-truth training jobs over the entire population after search
        submit_gt_jobs(args, max_step=40000, start_config=0, bundle_count=20,
                       n_gpus=8, gpu_config='dgx1_8gpu_fp32', targets=['NLX-NDv2'])

    elif args['phase'] == 'extract_pareto':
        # --------------- extract proxy pareto from all samples seen during the evolutionary search
        ppl_eps = 1  # abosulte ppl difference for extracting the pareto
        param_eps = 0.01  # nomarlized parameter diff for extracting the pareto
        eps = ppl_eps if (args['start_train'] < args['n_iter'] and hybrid) else param_eps
        get_final_pareto_front(args, eps=eps, hybrid=hybrid, use_convex_hull=use_convex_hull)

    elif args['phase'] == 'select_pareto':
        # --------------- match proxy pareto front points with the baseline and submit selected points on the pareto front for full training
        path_to_baseline = os.path.join(path_to_amlt_results, 'evolution_baselines')

        with open(os.path.join(path_to_baseline, 'params.pkl'), 'rb') as f:
            params_baseline = pickle.load(f)

        with open(os.path.join(path_to_baseline, 'latency_summary_{}.yaml'.format(args['device_name'])), 'r') as f:
            latency_baseline = yaml.safe_load(f)

        baseline_params_list = []
        baseline_latency_list = []

        for k, p in params_baseline.items():
            baseline_params_list.append(p)
            baseline_latency_list.append(latency_baseline[k])

        fname = 'pareto{}'.format('' if hybrid else '_params')
        fname += '_convexHull' if use_convex_hull else ''

        path_to_logs = os.path.join(args['results_path'], fname+'_points.pkl')
        with open(path_to_logs, 'rb') as f:
            pareto = pickle.load(f)

        indices = set()
        keys_to_keep = set()

        for l_b, p_b in zip(baseline_latency_list, baseline_params_list):
            candidate_param = 0
            candidate_latency = np.Inf

            index_param = None
            index_latency = None

            for i, (l, p) in enumerate(zip(pareto['latencies'], pareto['params'])):
                if abs(p-p_b) < 0.01*p_b and l < candidate_latency and l < l_b:
                    index_latency = i
                    candidate_latency = l

                if abs(l-l_b) < 0.01*l_b and p > candidate_param and p > p_b:
                    index_param = i
                    candidate_param = p

            if index_param is not None:
                indices.add(index_param)
                keys_to_keep.add(pareto['keys'][index_param])

            if index_latency is not None:
                indices.add(index_latency)
                keys_to_keep.add(pareto['keys'][index_latency])

        indices = list(indices)

        x_pareto = np.asarray(pareto['params'])
        y_pareto = np.asarray(pareto['latencies']) * 1000.

        plt.figure()
        plt.scatter(x_pareto[indices], y_pareto[indices], s=5, label=k)
        plt.scatter(baseline_params_list, np.asarray(baseline_latency_list)*1000., s=5, label='baseline')
        plt.ylabel('Latency (ms)')
        plt.xlabel('Decoder nParams')
        plt.title('Pareto Curve')
        plt.grid(axis='y')
        plt.legend()
        plt.savefig(os.path.join(args['results_path'], 'search_paretos{}.png'.format('' if hybrid else '_params')), bbox_inches="tight")

        path_to_logs = os.path.join(args['results_path'], fname+'.pkl')
        with open(path_to_logs, 'rb') as f:
            is_pareto_dict = pickle.load(f)

        ks = list(is_pareto_dict.keys())
        for k in ks:
            if not k in keys_to_keep:
                is_pareto_dict[k] = False

        submit_pareto_front_jobs(args, is_pareto_dict, max_step=40000, start_config=0,
                                 bundle_count=20, n_gpus=8, gpu_config='dgx1_8gpu_fp32', targets=['NLX-NDv2'])

    elif args['phase'] == 'comp_w_baseline':
        # --------------- compare baseline with pareto NAS results
        training_exp_name = 'pareto_{}'.format(args['device_name'])
        compare_w_baseline(args, exp_name=training_exp_name, path_to_dir=path_to_amlt_results,
                           start_config=0, baseline_exp='evolution_baselines', check_pareto=False)

    elif args['phase'] == 'gt_pareto':
        # --------------- compare ground-truth pareto with the proxy pareto
        gt_exp_name = 'evolution_40000'
        os.makedirs(path_to_amlt_results, exist_ok=True)

        command = 'amlt results {} -I "*.json"  -o {} --no-md5'.format(gt_exp_name, path_to_amlt_results)
        os.system(command)
        
        command = 'amlt results {} -I "*.yaml"  -o {} --no-md5'.format(gt_exp_name, path_to_amlt_results)
        os.system(command)
        
        get_gt_pareto(args, exp_name=gt_exp_name, path_to_dir=path_to_amlt_results, start_config=0,
                      ppl_eps=0.1, latency_eps=0.01, hybrid=hybrid, use_convex_hull=use_convex_hull, min_acceptable_latency_diff=2, baseline_exp=None)  # 'evolution_baselines')
        
        compare_w_baseline(args, exp_name=gt_exp_name, path_to_dir=path_to_amlt_results,
                           start_config=0, baseline_exp='evolution_baselines', check_pareto=True)

        # ---------------- compare final val ppl with nparams pareto
        with open('amlt_logs/evolution_40000/params_summary.yaml') as f:
            n_all_params = yaml.load(f)

        gt_results = get_results('evolution_40000', 'amlt_logs/evolution_40000', filetypes=['.json'], verbose=False)

        params_list = []
        val_ppl_list = []

        for job_name, result in gt_results.items():
            if job_name in n_all_params.keys():
                params_list.append(n_all_params[job_name]['FFN'] + n_all_params[job_name]['Attn'])
                val_ppl_list.append(result['valid_perplexity'])

        max_ppl_diff = 0.
        for idx, p in enumerate(params_list):
            for idx2, p2 in enumerate(params_list):
                if abs(p-p2)*100./p < 0.1:
                    if abs(val_ppl_list[idx] - val_ppl_list[idx2]) > max_ppl_diff:
                        max_ppl_diff = abs(val_ppl_list[idx] - val_ppl_list[idx2])
                        max_idx = idx
        
        print(f'maximum vertical difference in val ppl={max_ppl_diff}, happend in p={params_list[idx]}')

        plt.figure()
        plt.scatter(params_list, val_ppl_list, s=5)
        plt.xlabel('# Decoder Params')
        plt.ylabel('Val PPL')
        plt.title('Pareto Curve')
        plt.grid(axis='y')
        fname = 'pareto_params.png'
        plt.savefig(os.path.join(args['results_path'], fname), bbox_inches="tight")
