import os
from pickle import TRUE
import pickle
import numpy as np
import yaml
import collections
import re
import copy
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import torch

def meta_constructor_mapping(loader, node):
    value = loader.construct_mapping(node)
    return value

def meta_constructor_sequence(loader, node):
    value = loader.construct_sequence(node)
    return value

yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar', meta_constructor_sequence)
yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.dtype', meta_constructor_mapping)


def get_config_name(job):
  idx =  re.search('(config_[0-9]+)', job).span()[0]
  job = job[idx:]
  config_name = job.split('/')[0]
  return config_name + '_' + job.split('/')[1]
  

def recurse_dir(path_to_dir):
  results = {}
  for j in os.listdir(path_to_dir):
      j_path = os.path.join(path_to_dir, j)
      if os.path.isdir(j_path):
        results.update(recurse_dir(j_path))
      else:
        config = None
        if os.path.basename(j_path) == 'config.yaml':
          with open(os.path.join(j_path), 'r') as f:
            config = yaml.safe_load(f)

          if config:   
            config_name = get_config_name(j_path)
            print(config_name)
            results[config_name] = config
  
  return results


def profile_baseline(evolution_obj, path_to_results):
  configs = recurse_dir(path_to_results)
  
  proxies = {}
  total_params = {}
  latencies = {}
  memories = {}
  logs = {'configs': [],
          'proxies': [],
          'total_params': [],
          'latencies': [],
          'memories': []}

  for job_name, model_config in configs.items():
    gene = evolution_obj.converter.config_to_gene(model_config)
    config_, proxy, total_param, latency, memory = evolution_obj._calculate_gene_constraints(gene)
    print(f'''{job_name} has: 
            {proxy} {evolution_obj.constraint_strategy}
            {total_param} total_params
            {latency:.4f}s latency
            {memory:.4f}MB memory''')

    proxies[job_name] = proxy
    total_params[job_name] = total_param
    latencies[job_name] = latency
    memories[job_name] = memory

    logs['configs'].append(config_)
    logs['proxies'].append(proxy)
    logs['total_params'].append(total_param)
    logs['latencies'].append(latency)
    logs['memories'].append(memory)

  logs_path = os.path.join(path_to_results, f'logs.pkl')
  with open(logs_path, 'wb') as f:
      pickle.dump({'configs': logs['configs'],
                    'proxies': logs['proxies'],
                    'total_params': logs['total_params'],
                    'latencies': logs['latencies'],
                    'memories': logs['memories']}, f)

  yaml_file = os.path.join(path_to_results, 'proxies_summary.yaml')
  with open(yaml_file, 'w') as f:
    yaml.dump(proxies, f)
  yaml_file = os.path.join(path_to_results, 'total_params_summary.yaml')
  with open(yaml_file, 'w') as f:
    yaml.dump(total_params, f)
  yaml_file = os.path.join(path_to_results, 'latencies_summary.yaml')
  with open(yaml_file, 'w') as f:
    yaml.dump(latencies, f)
  yaml_file = os.path.join(path_to_results, 'memories_summary.yaml')
  with open(yaml_file, 'w') as f:
    yaml.dump(memories, f)

  return logs

      
def select_pareto(evolution_obj, path_to_results):
  with open(os.path.join(path_to_results, 'logs.pkl'), 'rb') as f:
    baseline = pickle.load(f) 
  
  indices = set()
  for p_b, l_b, m_b in zip(baseline['proxies'], baseline['latencies'], baseline['memories']):
    if evolution_obj.constraint_strategy == 'decoder_params':
      candidate_proxy = 0 
    else:
      raise NotImplementedError
    candidate_latency = np.Inf
    candidate_memory = np.Inf
    
    index_proxy = None
    index_latency = None
    index_memory = None
    for i, (p, l, m) in enumerate(zip(evolution_obj.pareto['proxies'], evolution_obj.pareto['latencies'], evolution_obj.pareto['memories'])):
      if l == np.min(evolution_obj.pareto['latencies']) or m == np.min(evolution_obj.pareto['memories']):
        indices.add(i)
      
      if abs(p-p_b) < 0.05*p_b:
        if l < l_b and l < candidate_latency:
          index_latency = i
          candidate_latency = l
        if m < m_b and m < candidate_memory:
          index_memory = i
          candidate_memory = m

      if abs(l-l_b) < 0.05*l_b: 
        if p > p_b and p > candidate_proxy:
          index_proxy = i
          candidate_proxy = p
        if m < m_b and m < candidate_memory:
          index_memory = i
          candidate_memory = m

      if abs(m-m_b) < 0.05*m_b: 
        if p > p_b and p > candidate_proxy:
          index_proxy = i
          candidate_proxy = p
        if l < l_b and l < candidate_latency:
          index_latency = i
          candidate_latency = l
      
    if index_proxy is not None:
      indices.add(index_proxy)

    if index_latency is not None:
      indices.add(index_latency)

    if index_memory is not None:
      indices.add(index_memory)
  
  indices = list(indices)
  print(f'Selected {len(indices)} of pareto jobs based on their proximity to the baseline architectures.')
  
  evolution_obj.pareto = {'population': [evolution_obj.pareto['population'][i] for i in indices],
                       'proxies': [evolution_obj.pareto['proxies'][i] for i in indices],
                       'total_params': [evolution_obj.pareto['total_params'][i] for i in indices],
                       'latencies': [evolution_obj.pareto['latencies'][i] for i in indices],
                       'memories': [evolution_obj.pareto['memories'][i] for i in indices]}

  evolution_obj.plot_search_state(iteration=None, parents=None, baseline=baseline)
  
  
  baseline['pareto'] = [evolution_obj.pareto]
  with open(os.path.join(path_to_results, 'logs.pkl'), 'wb') as f:
    pickle.dump(baseline, f)
  
  return evolution_obj