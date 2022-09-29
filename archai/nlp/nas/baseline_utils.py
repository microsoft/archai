import collections
import os
from pickle import TRUE
import pickle
import numpy as np
from tqdm import trange
import yaml
import json
import re
import copy
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

from archai.nlp.nas.nas_utils.plotter import plot_2d_pareto, plot_3d_pareto

def meta_constructor_mapping(loader, node):
    value = loader.construct_mapping(node)
    return value

def meta_constructor_sequence(loader, node):
    value = loader.construct_sequence(node)
    return value

yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar', meta_constructor_sequence)
yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.dtype', meta_constructor_mapping)


def get_indices_to_keep(x_pareto, y_pareto, threshold=0.005):
  # x_pareto lower is better
  # y_pareto lower is better

  indices_to_keep = []
  for i, (x1, y1) in enumerate(zip(x_pareto, y_pareto)):
    is_pareto = True
    for j,(x2, y2) in enumerate(zip(x_pareto, y_pareto)):
      if i==j:
        continue
      if y2 < y1 and x2 <= x1:
        if (y1-y2) > threshold * y2:
          is_pareto = False
          break
    if is_pareto:
      indices_to_keep.append(i)

  return indices_to_keep
  

def plot_paper(x_pareto, y_pareto, x_baseline, y_baseline, x_label, y_label, path_to_save, indices_to_keep, scale_x=1):
  x_pareto, y_pareto = x_pareto[indices_to_keep], y_pareto[indices_to_keep]

  indices = np.argsort(x_pareto)
  x_pareto, y_pareto = x_pareto[indices], y_pareto[indices]
  indices = np.argsort(x_baseline)
  x_baseline, y_baseline = x_baseline[indices], y_baseline[indices]

  plt.figure(figsize=(5,3))
  plt.plot(np.asarray(x_pareto) * scale_x, y_pareto, markersize=10, label='LTS', color='midnightblue', marker='.')
  plt.plot(np.asarray(x_baseline) * scale_x, y_baseline, markersize=5, label='Scaled Transformer', color='tab:blue', marker='d')
  # plt.xlim((min(np.min(x_pareto), np.min(x_baseline))*scale_x-10, np.max(gt_latencies)*1000+10))
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.grid(axis='y')
  plt.legend(handletextpad=0.1, borderpad=0)
  plt.savefig(path_to_save, bbox_inches="tight")


  print('baseline points:', [(v, l) for v, l in zip(y_baseline, x_baseline)])
  print('pareto points:', [(v, l) for v, l in zip(y_pareto, x_pareto)])

  diff = []
  for (x, y) in zip(x_pareto, y_pareto):
    # find closest baseline point
    idx = np.argmin(np.absolute(x_baseline - x))
    if abs(x_baseline[idx] - x) < 0.05*x:
      diff.append((y_baseline[idx] - y)*100./y)
  print('on {}/{} baseline models, for the same metric we get {}%% lower ppl on average.'.format(len(diff), len(x_pareto), np.mean(diff)))


def get_config_name(job):
  idx =  re.search('(config_[0-9]+)', job).span()[0]
  job = job[idx:]
  config_name = job.split('/')[0]
  return config_name + '_' + job.split('/')[1]
  

def get_info_from_json(json_file, metric=['valid_perplexity', 'valid_ppl']):
  '''
    step: step number to extract the ppl log, live empty to get the final ppl 
    metric: type of metric to read from the json file
  '''
  out_dict = {}
  with open(json_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()[::-1]
    try:
      job_desc = re.search('DLLL \{(.+?)\n', lines[-1])
    except:
      return None
    job_desc = '{'+job_desc.group(1)
    work_dir = json.loads(job_desc)['data']['work_dir']
    try:
      idx_start = re.search('amlt-results', work_dir).span()[-1] + 1
      amlt_job = work_dir[idx_start:].split('/')[0]
    except:
      amlt_job = None
  
    for line in lines:
      str = re.search('DLLL \{(.+?)\}', line)
      str = '{'+str.group(1)+'}}'
      final_train_log = json.loads(str)
      try:
        out_dict['train_elapsed'] = float(final_train_log['data']['train_elapsed'])*60
        for k in final_train_log['data'].keys():
          if k in metric:
            out_dict[k] = final_train_log['data'][k]

        out_dict['amlt_job'] = amlt_job
        break
      except:
        return None
  
  return out_dict


def recurse_dir(path_to_dir, fname='config.yaml'):
  results = {}
  for j in os.listdir(path_to_dir):
      j_path = os.path.join(path_to_dir, j)
      if os.path.isdir(j_path):
        results.update(recurse_dir(j_path, fname=fname))
      else:
        config = None
        if os.path.basename(j_path) == fname:
          if '.yaml' in fname:
            with open(os.path.join(j_path), 'r') as f:
              config = yaml.safe_load(f)
          elif '.json' in fname:
            config = get_info_from_json(os.path.join(j_path))
          else:
            raise NotImplementedError

        if config is None and '.yaml' in fname:
          try:
            json_file = os.path.join(path_to_dir, 'train_log.json')
            with open(json_file, 'r', encoding='utf-8') as f:
              lines = f.readlines()
              try:
                job_desc = re.search('DLLL \{(.+?)\}', lines[0])
              except:
                return None
              job_desc = '{'+job_desc.group(1)+'}}'
              config = json.loads(job_desc)['data']
          except:
            continue

        if config:   
          config_name = get_config_name(j_path)
          results[config_name] = config
  
  return results


def config_to_key(config, name=None, keys=['n_layer', 'd_model', 'd_inner','n_head', 'div_val']):
  short_config = {}
  for k in keys:
    if isinstance(config[k], list) and len(config[k])==1:
      short_config[k] = config[k][0]
    else:
      short_config[k] = config[k]
  if name is not None:
    short_config['name'] = name
  return short_config


def profile_baseline(evolution_obj, path_to_results, device_name, dataset):
  training_path = os.path.join(path_to_results, dataset)
  path_to_configs = os.path.join(training_path, 'model_configs.yaml')
  if os.path.exists(path_to_configs):
    with open(path_to_configs, 'r') as f:
      configs = yaml.safe_load(f)
  else:
    fname = 'config.yaml' if evolution_obj.model_type == 'mem_transformer' else 'model_config.yaml'
    configs = recurse_dir(training_path, fname=fname)
    with open(path_to_configs, 'w') as f:
      yaml.dump(configs, f)
  
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

  logs_path = os.path.join(path_to_results, device_name)
  with open(os.path.join(logs_path, 'logs.pkl'), 'wb') as f:
      pickle.dump({'configs': logs['configs'],
                    'proxies': logs['proxies'],
                    'total_params': logs['total_params'],
                    'latencies': logs['latencies'],
                    'memories': logs['memories']}, f)

  yaml_file = os.path.join(logs_path, 'proxies_summary.yaml')
  with open(yaml_file, 'w') as f:
    yaml.dump(proxies, f)
  yaml_file = os.path.join(logs_path, 'total_params_summary.yaml')
  with open(yaml_file, 'w') as f:
    yaml.dump(total_params, f)
  yaml_file = os.path.join(logs_path, 'latencies_summary.yaml')
  with open(yaml_file, 'w') as f:
    yaml.dump(latencies, f)
  yaml_file = os.path.join(logs_path, 'memories_summary.yaml')
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
  
  model_configs = []
  for gene in evolution_obj.pareto['population']:
    config = evolution_obj.converter.gene_to_config(gene)   
    curr_model_config = copy.deepcopy(evolution_obj.model_config)
    curr_model_config.update(config)
    curr_model_config = curr_model_config.to_dict()
    model_configs.append(curr_model_config)
  
  evolution_obj.pareto['model_configs'] = model_configs

  baseline['pareto'] = [evolution_obj.pareto]
  with open(os.path.join(path_to_results, 'logs.pkl'), 'wb') as f:
    pickle.dump(baseline, f)
  
  return evolution_obj


def plot_baseline_and_pareto(path_to_amlt_logs, path_to_baseline_logs, dataset, device_name): 
  # load all info for baseline models
  baseline_logs = recurse_dir(os.path.join(path_to_baseline_logs, dataset), fname='train_log.json')    # load baseline val_ppls
  baseline_configs = recurse_dir(os.path.join(path_to_baseline_logs, dataset), fname='model_config.yaml')    # load baseline model configs

  with open(os.path.join(os.path.join(path_to_baseline_logs, device_name), 'latencies_summary.yaml'), 'r') as f:   # load baseline latencies
    baseline_latencies = yaml.safe_load(f)
  with open(os.path.join(os.path.join(path_to_baseline_logs, device_name), 'memories_summary.yaml'), 'r') as f:    # load baseline memories
    baseline_memories = yaml.safe_load(f)
  with open(os.path.join(os.path.join(path_to_baseline_logs, device_name), 'proxies_summary.yaml'), 'r') as f:    # load baseline memories
    baseline_params = yaml.safe_load(f)
  
  # load all info for (selected) pareto models
  pareto_train_logs = recurse_dir(path_to_amlt_logs, fname='train_log.json')   # load pareto val_ppls
  pareto_configs = recurse_dir(path_to_amlt_logs, fname='model_config.yaml')         # load pareto model configs

  fname = 'logs.pkl'
  with open(os.path.join(path_to_baseline_logs, device_name, fname), 'rb') as f:       # load pareto memories and latencies
    pareto_logs = pickle.load(f)['pareto'][0]

  # with open('logdir/mem_transformer_titanxp_3d/logs_iter_29.pkl', 'rb') as f:
  #   pareto_ = pickle.load(f)
  # match_indices = collections.OrderedDict({})
  # for config_idx in range(4):
  #   for job_idx in range(10):
  #     if f'config_{config_idx}_j0' not in pareto_train_logs.keys():
  #       continue
  #     for idx in range(len(pareto_logs['model_configs'])):
  #       match = True
  #       this_config = config_to_key(pareto_configs[f'config_{config_idx}_j{job_idx}'])
  #       orig_config = config_to_key(pareto_logs['model_configs'][idx])
  #       for k, v in this_config.items():
  #         if v != orig_config[k]:
  #           match = False
  #           break
        
  #       if match:
  #         found = True
  #         break
  #     assert found
  #     print('idx {} matched to {}'.format(idx, f'config_{config_idx}_j{job_idx}'))
  #     match_indices[f'config_{config_idx}_j{job_idx}'] = idx
  # keys = list(pareto_logs.keys())
  # for k in keys:
  #   print('reordering ', k)
  #   pareto_logs[k] = (np.asarray(pareto_logs[k])[list(match_indices.values())]).tolist()
  # with open(os.path.join(path_to_baseline_logs, device_name, fname), 'rb') as f:       # load pareto memories and latencies
  #   all_logs = pickle.load(f)['pareto'][0]
  # all_logs['pareto'] = [pareto_logs]
  # with open(os.path.join(path_to_baseline_logs, device_name, fname), 'wb') as f:       # load pareto memories and latencies
  #   pickle.dump(all_logs, f)

  all_val_ppls = []
  all_configs = []
  all_latencies = []
  all_memories = []
  # config_idx, job_idx, idx = 0, 0, 0
  n_configs = 4
  n_jobs_perconfig = np.ceil(len(pareto_logs['model_configs']) / n_configs)
  for idx in range(len(pareto_logs['model_configs'])):
    config_idx = int(idx // n_jobs_perconfig)
    job_idx = int(idx % n_jobs_perconfig)
    # if f'config_{config_idx}_j0' not in pareto_train_logs.keys():
    #   break
    # while True:
    #   if f'config_{config_idx}_j{job_idx}' not in pareto_train_logs.keys():
    #     break
    if f'config_{config_idx}_j{job_idx}' not in pareto_train_logs.keys():
      continue
    print(f'config_{config_idx}_j{job_idx}')

    # make sure these are the same models
    this_config = config_to_key(pareto_configs[f'config_{config_idx}_j{job_idx}'])
    orig_config = config_to_key(pareto_logs['model_configs'][idx])
    for k, v in this_config.items():
      assert v == orig_config[k], print(f'config_{config_idx}_j{job_idx}', k, v, orig_config[k])
    l, m = pareto_logs['latencies'][idx], pareto_logs['memories'][idx]

    all_val_ppls.append(pareto_train_logs[f'config_{config_idx}_j{job_idx}']['valid_ppl'])
    all_configs.append(config_to_key(pareto_configs[f'config_{config_idx}_j{job_idx}'], name=f'config_{config_idx}_j{job_idx}'))
    all_latencies.append(l)
    all_memories.append(m)
    
    # job_idx += 1
    # idx += 1
    
    # config_idx += 1
    # job_idx = 0

  all_val_ppls = np.asarray(all_val_ppls)
  all_configs = np.asarray(all_configs)
  all_latencies = np.asarray(all_latencies)
  all_memories = np.asarray(all_memories)

  job_keys = np.sort(list(baseline_memories.keys()))
  try:
    baseline_val_ppls = np.asarray([baseline_logs[k]['valid_perplexity'] for k in job_keys])
  except:
    baseline_val_ppls = np.asarray([baseline_logs[k]['valid_ppl'] for k in job_keys])
  baseline_configs = np.asarray([config_to_key(baseline_configs[k], name=k) for k in job_keys])
  baseline_latencies = np.asarray([baseline_latencies[k] for k in job_keys])
  baseline_memories = np.asarray([baseline_memories[k] for k in job_keys])

  indices_latency = get_indices_to_keep(x_pareto=all_latencies, y_pareto=all_val_ppls)
  indices_memory = get_indices_to_keep(x_pareto=all_memories, y_pareto=all_val_ppls)
  indices_to_keep = np.union1d(indices_latency, indices_memory)
  print(f'keeping {len(indices_to_keep)} pareto models')
  with open(os.path.join(path_to_baseline_logs, device_name, f'indices_to_keep_{dataset}.pkl'), 'wb') as f:
    pickle.dump(indices_to_keep, f)

  lat_diff = []
  mem_diff = []
  for l, m , ppl in zip(baseline_latencies, baseline_memories, baseline_val_ppls):
    idx = np.argmin(np.absolute(all_val_ppls - ppl))
    if idx in indices_to_keep and np.absolute(all_val_ppls[idx] - ppl) < 0.1 * ppl:
      lat_diff.append((l - all_latencies[idx])*100./l)
      mem_diff.append((m - all_memories[idx])*100./m)

  print('on {}/{} baseline models, for the same ppl we get less {}%% latency and {}%% memory on average.'.format(len(baseline_val_ppls), len(lat_diff), np.mean(lat_diff), np.mean(mem_diff)))

  # 2D plot: val_ppl x latencies 
  visited_dict = {'x': all_val_ppls, 'y': all_latencies, 'config': all_configs}
  pareto_dict = visited_dict
  baseline_dict = {'x': baseline_val_ppls, 'y': baseline_latencies, 'config': baseline_configs}
  output_path = os.path.join(path_to_amlt_logs, f'val_ppl_vs_latency')

  plot_2d_pareto(visited_dict,
                  pareto_dict,
                  parents=None,
                  baseline=baseline_dict,
                  hover_template='Val ppl: %{x:.2f}' + '<br>Latency (s): %{y:.4f}<br>' + '%{text}',
                  title_text=f'Val ppl vs. Latency (s)',
                  xaxis_title='Val ppl',
                  yaxis_title='Latency (s)',
                  output_path=output_path) 
  print('***************************** latency *****************************')
  plot_paper(x_pareto=all_latencies, y_pareto=all_val_ppls, x_baseline=baseline_latencies, y_baseline=baseline_val_ppls, 
            x_label='Latency (ms)', y_label='Validation PPL', path_to_save=output_path, scale_x=1000., indices_to_keep=indices_latency)
  
  # 2D plot:  val_ppl x memories 
  visited_dict = {'x': all_val_ppls, 'y': all_memories, 'config': all_configs}
  pareto_dict = visited_dict
  baseline_dict = {'x': baseline_val_ppls, 'y': baseline_memories, 'config': baseline_configs}
  output_path = os.path.join(path_to_amlt_logs, f'val_ppl_vs_memory')

  plot_2d_pareto(visited_dict,
                  pareto_dict,
                  parents=None,
                  baseline=baseline_dict,
                  hover_template='Val ppl: %{x:.2f}' + '<br>Memory (MB): %{y:.4f}<br>' + '%{text}',
                  title_text=f'Val ppl vs. Memory (MB)',
                  xaxis_title='Val ppl',
                  yaxis_title='Memory (MB)',
                  output_path=output_path)
  print('***************************** memory *****************************')
  plot_paper(x_pareto=all_memories, y_pareto=all_val_ppls, x_baseline=baseline_memories, y_baseline=baseline_val_ppls, 
            x_label='Memory (MB)', y_label='Validation PPL', path_to_save=output_path, scale_x=1., indices_to_keep=indices_memory)

  # 3D plot: val_ppl x latencies x memories 
  visited_dict = {'x': all_val_ppls, 'y': all_memories, 'z': all_latencies, 'config': all_configs}
  pareto_dict = visited_dict
  baseline_dict = {'x': baseline_val_ppls, 'y': baseline_memories, 'z': baseline_latencies, 'config': baseline_configs}
  output_path = os.path.join(path_to_amlt_logs, f'val_ppl_vs_memory_vs_latency')

  plot_3d_pareto(visited_dict,
                  pareto_dict,
                  parents=None,
                  baseline=baseline_dict,
                  hover_template='Val ppl: %{x:.2f}' + '<br>Memory (MB): %{y:.4f}<br>' + 'Latency (s): %{z:.4f}<br>' + '%{text}',
                  title_text=f'Val ppl vs. Memory (MB) vs. Latency (s)',
                  xaxis_title='Val ppl',
                  yaxis_title='Memory (MB)',
                  zaxis_title='Latency (s)',
                  output_path=output_path)


def get_latex_tables(path_to_baseline_logs, dataset, device_name):
    fname = 'logs.pkl'
    with open(os.path.join(path_to_baseline_logs, device_name, fname), 'rb') as f:       # load pareto memories and latencies
      pareto_logs = pickle.load(f)['pareto'][0]

    filename = 'table_{}.tex'.format(device_name)
    with open(filename, 'w') as f:
        f.write('\\begin{table*}[h]\n'+
                '\\resizebox{\\textwidth}{!}{\n'+
                '\\begin{tabular}{lcccccc} \hline\n'+
                '\multicolumn{2}{c}{}\n'+         
                '& n\\textsubscript{layer} \n'+
                '& d\\textsubscript{model} \n'+
                '& n\\textsubscript{head} \n'+
                '& d\\textsubscript{inner}\n'+
                '& DecoderParams (M) \\\\ \\hline \n'+
                '\\multicolumn{2}{l}{baseline} & $\in$[1,16] & 8 & 2048 & 512 & - \\\\ \\hline\n')
        f.write('\multirow{%s}{*}{\\rotatebox{90}{%s}}' % (len(pareto_logs['model_configs']), device_name))
    
    # with open(os.path.join(path_to_baseline_logs, device_name, f'indices_to_keep_{dataset}.pkl'), 'rb') as f:
    #   indices_to_keep = pickle.load(f)
    indices_to_keep = range(len(pareto_logs['model_configs']))

    table_configs = []
    params = []
    for idx in range(len(pareto_logs['model_configs'])):
        if not idx in indices_to_keep:
          continue
        this_config = config_to_key(pareto_logs['model_configs'][idx])
        nonemb_params = pareto_logs['proxies'][idx]
    
        table_configs.append('&{} &{} &{} &{} &{:.1f} \\\\ \n'.format(this_config['n_layer'],
                                    this_config['d_model'], this_config['n_head'], this_config['d_inner'],
                                    nonemb_params/1e6))
        params.append(nonemb_params)

    indices = np.argsort(params)
    for i, idx in enumerate(indices):
        with open(filename, 'a+') as f:
            f.write('&M{} '.format(i+1) + table_configs[idx])
    
    with open(filename, 'a+') as f:
        f.write('\\hline\n' + '\\end{tabular}}\n' + '\\end{table*}')
    
    exit()