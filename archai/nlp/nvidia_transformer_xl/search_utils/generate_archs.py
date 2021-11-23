import os
import numpy as np
import collections
import yaml
import math
import pprint
import random
import collections
import re
import copy
import json

from archai.nlp.nvidia_transformer_xl.mem_transformer import MemTransformerLM, MemTransformerLM_flex
from archai.nlp.nvidia_transformer_xl.search_utils.utils import recurse_dir, get_model_and_params
from archai.nlp.nvidia_transformer_xl.search_utils.gather_results import get_config_name, get_info_from_json, get_info_from_logs, process_parameters

_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

def meta_constructor_mapping(loader, node):
    value = loader.construct_mapping(node)
    return value

def meta_constructor_sequence(loader, node):
    value = loader.construct_sequence(node)
    return value

yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar', meta_constructor_sequence)
yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.dtype', meta_constructor_mapping)

generation_seed = 4568
np.random.seed(generation_seed)
random.seed(generation_seed)

phase = 5   # 1: submit jobs (optionally with fear stage 1 activated), 2: fear stage 2, 3: baseline, 4: similar parameter exploration, 5: transformer-XL baseline
activate_fear = False
n_unfreeze = 3  # used in phase 2
different_seeds = None #[1111,1009,1200,1234,1302,1562,2222,3334,3425,4567]
max_step = 500

targets = ['itpeastusv100cl', 'itplabrr1cl1', 'itpscusv100cl', 'itpseasiav100cl']  # amulet targets
gpu_config = 'dgx1_4gpu_fp32' # dgx1_8gpu_fp16, dgx1_1gpu_fp16, toy, default, dgx1_4gpu_fp16
n_gpus = 4
bundle_count = 4

n_configs = 100 # number fo configs to generate
batch_configs = 50 #how many configs in the same bash file
start_config = 0 #index for the starting job name, careful not to override previous jobs in the same experiment
bash_start_config = start_config
indep_dhead = False # if set to False, d_head is determined based on n_head and d_model so that n_head * d_head = d_model
homogeneous_layers = False # if set to True, all decoder layers will have the same config within a model

n_layers = [4, 8]
n_heads = [2, 8]
d_models = [64, 512]
#-----embedding layer
d_embeds = [128, 512]
div_vals = [1, 2, 4]
#-----optional
d_heads = [16, 64]
d_inners = [512, 2048]

model_config_keys = ['n_token', 'n_layer','n_head','d_model','d_head','d_inner','dropout','dropatt', \
                        'd_embed','div_val','pre_lnorm','tgt_len','ext_len','mem_len', \
                        'same_length','attn_type','clamp_len','sample_softmax']


#TODO: add this to fear stage 2 and 3
def generate_bash_files(path_to_configs, f_name, bash_start=0, exp_name=None):
  assert exp_name is not None, 'provide an experiment name for amulet jobs'
  
  bash_idx = 0
  while True:
    for t in range(len(targets)):
      for i in range(batch_configs):
        job_idx = i + t * batch_configs + bash_idx * len(targets) * batch_configs
        print(job_idx)
        if job_idx >= n_configs:
          break
        
        bash_file = os.path.join(path_to_configs, f_name+'_'+str(bash_idx+bash_start)+'.sh')
        if job_idx==0 and os.path.exists(bash_file):
              os.remove(bash_file)  
        with open(bash_file, 'a') as f:
          f.write('amlt run --yes archai/nlp/nvidia_transformer_xl/configs/{} {} -t {}\n'.format('nv_train_{}.yaml'.format(job_idx+bash_start_config), exp_name, targets[t]))
        
        # if different_seeds:
        #   f.write('amlt run --yes archai/nlp/nvidia_transformer_xl/configs/{} {} -t {}\n'.format('nv_train_{}.yaml'.format(job_idx+start_config), exp_name, targets[t]))
        # else:
        #   f.write('amlt run --yes archai/nlp/nvidia_transformer_xl/configs/{} {} -t {}\n'.format('nv_train_{}.yaml'.format(job_idx+start_config), exp_name, targets[t]))          
      if job_idx >= n_configs:
          break      
    if job_idx >= n_configs:
      break

    bash_idx += 1
  
  return (bash_idx + bash_start + 1)


def gather_results(exp_name, path_to_dir, filetypes='.json', verbose=True):
  if not isinstance(filetypes, list):
    filetypes = [filetypes]
  
  results = {}
  for j in os.listdir(path_to_dir):
      j_path = os.path.join(path_to_dir, j)
      if os.path.isdir(j_path):
        results.update(gather_results(exp_name, j_path, filetypes, verbose))
      else:
        for ft in filetypes:
          if ft in j:
            if '.log' in ft:
              logs = get_info_from_logs(j_path, stage_1='stage_1' in exp_name)
            elif '.json' in ft:
              logs = get_info_from_json(j_path)
            elif '.yaml' in ft:
              with open(os.path.join(j_path), 'r') as f:
                config = yaml.load(f)
              if config is None:
                  json_file = os.path.join(path_to_dir, 'train_log.json')
                  with open(json_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    try:
                      job_desc = re.search('DLLL \{(.+?)\}', lines[0])
                    except:
                      return None
                    job_desc = '{'+job_desc.group(1)+'}}'
                    config = json.loads(job_desc)['data']
                    config['n_token'] = 267735  
                
              model_config = {k: config[k] for k in model_config_keys}
                
              cutoffs, tie_projs = [], [False]
              if config['adaptive']:
                  assert config['dataset'] in ['wt103', 'wt2', 'lm1b']
                  if config['dataset'] in ['wt103', 'wt2']:
                      cutoffs = [19997, 39997, 199997]
                      tie_projs += [True] * len(cutoffs)
                  elif config['dataset'] == 'lm1b':
                      cutoffs = [59997, 99997, 639997]
                      tie_projs += [False] * len(cutoffs)
              model_config['cutoffs'] = cutoffs
              model_config['tie_projs'] = tie_projs
              model_config['tie_weight'] = config['tied']
              model_config['dtype'] = None
              logs = model_config
            else:
              assert False, 'unsupported file type {}'.format(ft)
            
            if logs: 
              config_name = get_config_name(j_path) #get_config_name(os.path.basename(os.path.dirname(j_path)))
              if verbose:
                print(config_name, logs)
              if config_name in results.keys():
                results[config_name].update(logs)
              else:
                results[config_name] = logs
  
  return results


def get_bundle_run_command(configs, parallel=True, default_config='wt103_base.yaml'):
  for k in configs.keys():
    configs[k] = [str(get_yaml_values(v)) for v in configs[k]]
  print(configs)

  command = '' if parallel else []
  master_ports = [1234, 2222, 2345, 1342]
  for i in range(len(configs['n_layer'])):
    gpu_range = ','.join([str(x) for x in range(i*n_gpus, (i+1)*n_gpus)])
    
    if parallel:
      command += 'export CUDA_VISIBLE_DEVICES=%s && \
                  python -m torch.distributed.launch --master_port=%d --nproc_per_node="%s" archai/nlp/nvidia_transformer_xl/train.py --config %s \
                  --config_file %s --n_layer %s --n_head %s --d_model %s --d_head %s \
                  --d_inner %s --d_embed %s --div_val %s --experiment_name j%d &\n' \
                  % (gpu_range, master_ports[i], str(n_gpus), gpu_config, default_config, configs['n_layer'][i], configs['n_head'][i], configs['d_model'][i], configs['d_head'][i], configs['d_inner'][i], configs['d_embed'][i], configs['div_val'][i], i)
    else:
      command.append('python -m torch.distributed.launch --nproc_per_node="%s" archai/nlp/nvidia_transformer_xl/train.py --config %s \
                  --config_file %s --n_layer %s --n_head %s --d_model %s --d_head %s \
                  --d_inner %s --d_embed %s --div_val %s --experiment_name j%d' \
                  % (str(n_gpus), gpu_config, default_config, configs['n_layer'][i], configs['n_head'][i], configs['d_model'][i], configs['d_head'][i], configs['d_inner'][i], configs['d_embed'][i], configs['div_val'][i], i))
  if parallel:
    command += 'wait'
  return command


def get_run_command(max_step, config_num, seed=None):
  if seed:
    assert config_num
    command = 'python -m torch.distributed.launch --nproc_per_node="%s" archai/nlp/nvidia_transformer_xl/train.py \
                                        --config {config} --config_file wt103_base.yaml \
                                        --n_layer {n_layer} --n_head {n_head} --d_model {d_model} --d_head {d_head} \
                                        --d_inner {d_inner} --d_embed {d_embed} --div_val {div_val} \
                                        --max_step %d --seed %d --experiment_name config_%s_seed_%d' % (str(n_gpus), max_step, seed, config_num, seed)
  else:
	  command = 'python -m torch.distributed.launch --nproc_per_node="%s" archai/nlp/nvidia_transformer_xl/train.py \
                                        --config {config} --config_file wt103_base.yaml \
                                        --n_layer {n_layer} --n_head {n_head} --d_model {d_model} --d_head {d_head} \
                                        --d_inner {d_inner} --d_embed {d_embed} --div_val {div_val} \
                                        --max_step %d --experiment_name config_%s_%d --scheduler constant' % (str(n_gpus), max_step, config_num, max_step)

  return command


def get_yaml_values(value):
  if isinstance(value, list):
    value_string = ''
    for v in value:
      value_string += (str(v) + ',')
    return value_string[:-1]
  else:
    return value


def build_dict(values):
  dict = {}
  if len(values)==2:
    dict = {values[0]: values[1]}
  else:
    dict[values[0]] = build_dict(values[1:])
  return dict


def parse_config(param_values, idx):
  config = {k:get_yaml_values(param_values[k][idx]) for k in param_values.keys()}
  return config


def find_duplicate(config, tree_all_configs):
  param_names = ['n_layer', 'n_head', 'd_model', 'd_head', 'd_inner', 'd_embed', 'div_val']

  curr_level = tree_all_configs
  not_in_tree = False
  for n in param_names[:-1]:
    if config[n] in curr_level.keys():
      curr_level = curr_level[config[n]]
    else:
      not_in_tree = True
      break
  
  if not_in_tree:
    param_values = [config[n] for n in param_names]
    dict_to_add = build_dict(param_values)
    tree_all_configs.update(dict_to_add)

  is_duplicate = not not_in_tree
  return is_duplicate, tree_all_configs


def generate_params_homogeneous(n_configs):
  # generate n_configs with homogeneous decoder layers 
  values = collections.OrderedDict({})
  values['n_layer'] = (np.random.randint(low=n_layers[0], high=n_layers[-1]+1, size=n_configs, dtype=np.int32)).tolist()
  values['n_head'] = (2**np.random.randint(low=np.log2(n_heads[0]), high=np.log2(n_heads[-1])+1, size=n_configs, dtype=np.int32)).tolist()
  values['d_model'] = (2**np.random.randint(low=np.log2(d_models[0]), high=np.log2(d_models[-1])+1, size=n_configs, dtype=np.int32)).tolist()
  # values['d_embed'] = (2**np.random.randint(low=np.log2(d_embeds[0]), high=np.log2(d_embeds[-1])+1, size=n_configs, dtype=np.int32)).tolist()
  values['d_embed'] = [2**np.random.randint(low=min(np.log2(values['d_model'][i]), np.log2(d_embeds[0])), high=min(np.log2(values['d_model'][i]), np.log2(d_embeds[-1]))+1, dtype=np.int32).tolist() for i in range(n_configs)]
  values['div_val'] = np.random.choice(div_vals, size=n_configs).astype(np.int32).tolist()
  
  if indep_dhead:
    values['d_head'] = (2**np.random.randint(low=np.log2(d_heads[0]), high=np.log2(d_heads[-1])+1, size=n_configs, dtype=np.int32)).tolist()
  else:
    values['d_head'] = [int(values['d_model'][i] / values['n_head'][i]) for i in range(n_configs)]
  
  # values['d_inner']    = (2**np.random.randint(low=np.log2(d_inners[0]), high=np.log2(d_inners[-1]), size=n_configs, dtype=np.int32)).tolist()
  values['d_inner'] = [random.randint(max(int(2*values['d_model'][i]), d_inners[0]), d_inners[-1]) for i in range(n_configs)]

  return values


def generate_params_heterogeneous(n_configs):
  # generate n_configs with heterogeneous decoder layers 
  values = collections.OrderedDict({})
  values['n_layer'] = (np.random.randint(low=n_layers[0], high=n_layers[-1]+1, size=n_configs, dtype=np.int32)).tolist()
  values['n_head'] = [(2**np.random.randint(low=np.log2(n_heads[0]), high=np.log2(n_heads[-1])+1, size=values['n_layer'][i], dtype=np.int32)).tolist() for i in range(n_configs)]
  values['d_model'] = (2**np.random.randint(low=np.log2(d_models[0]), high=np.log2(d_models[-1])+1, size=n_configs, dtype=np.int32)).tolist()
  # values['d_embed'] = (2**np.random.randint(low=np.log2(d_embeds[0]), high=np.log2(d_embeds[-1])+1, size=n_configs, dtype=np.int32)).tolist()
  values['d_embed'] = [2**np.random.randint(low=min(np.log2(values['d_model'][i]), np.log2(d_embeds[0])), high=min(np.log2(values['d_model'][i]), np.log2(d_embeds[-1]))+1, dtype=np.int32).tolist() for i in range(n_configs)]
  values['div_val'] = np.random.choice(div_vals, size=n_configs).astype(np.int32).tolist()

  if indep_dhead:
    values['d_head'] = [(2**np.random.randint(low=np.log2(0.5*(values['d_model'][i]//np.asarray(values['n_head'][i]))), high=np.log2(2*(values['d_model'][i]//np.asarray(values['n_head'][i])))+1, size=values['n_layer'][i], dtype=np.int32)).tolist() for i in range(n_configs)]
  else:
    values['d_head'] = [(values['d_model'][i]//np.asarray(values['n_head'][i])).tolist() for i in range(n_configs)]
  
  values['d_inner'] = [np.random.randint(low=max(int(2*values['d_model'][i]), d_inners[0]), high=d_inners[-1]+1, size=values['n_layer'][i]).tolist() for i in range(n_configs)]
  return values

  
def mismatch(value_1, value_2):
  if isinstance(value_1, list):
    value_1 = get_yaml_values(value_1)
  if value_1 != value_2:
    return True
  else:
    return False


def multiply(value, factor):
  if isinstance(value, list):
    return np.asarray([round(v * factor) for v in value]).astype(np.int32)
  else:
    return round(value * factor)


if __name__ == '__main__':
  path_to_configs = os.path.join('archai/nlp/nvidia_transformer_xl', 'configs')
  if not os.path.exists(path_to_configs):
      os.mkdir(path_to_configs)
  
  if phase==1:
      # generate random architectures
      count = 0
      param_values = {}
      # TODO: load tree_all_configs from file
      tree_all_configs = {}
      while count < n_configs:
        print('generating a new batch of configs')
        if homogeneous_layers:
          new_param_values = generate_params_homogeneous(n_configs)
          # pprint.pprint(new_param_values)
        else:
          new_param_values = generate_params_heterogeneous(n_configs)

        for idx in range(len(new_param_values['n_layer'])):
          config = parse_config(new_param_values, idx)
          is_duplicate, tree_all_configs = find_duplicate(config, tree_all_configs)
          if is_duplicate:
            print('duplicate config')
          else:
            for k in new_param_values.keys():
              if k in param_values.keys():
                param_values[k] += [new_param_values[k][idx]]
              else:
                param_values[k] = [new_param_values[k][idx]]
            count += 1
            if count==n_configs:
              break
      # pprint.pprint(param_values)
    
      # create corresponding yaml files for amulet jobs
      default_config = 'wt103_base_FEAR.yaml' if activate_fear else 'wt103_base.yaml'
      c = 0
      yaml_idx = start_config
      while c < n_configs: 
        with open('/home/t-mojanj/Projects/archaiphilly/nv_train.yaml') as file:
          amlt_config = yaml.load(file)
          if c==0:
            pprint.pprint(amlt_config)

        ''' # run one job per sku
        amlt_config['search']['job_template']['sku']= 'G'+str(n_gpus)
        amlt_config['search']['job_template']['name']= 'train_xl_config_'+str(c + start_config)
        # TODO: add vocab_size when there is support for it
        amlt_config['search']['job_template']['command'][3] = 'python -m torch.distributed.launch --nproc_per_node="%s" archai/nlp/nvidia_transformer_xl/train.py \
                                        --config {config} --config_file {default_config} \
                                        --n_layer {n_layer} --n_head {n_head} --d_model {d_model} --d_head {d_head} \
                                        --d_inner {d_inner} --d_embed {d_embed} --div_val {div_val}' % (str(n_gpus)) #--vocab_size {vocab_size} --attn_type 2 
        amlt_config['search']['params'][0]['values'] = gpu_config

        names = list(param_values.keys())   #['n_layer', 'n_head', 'd_model', 'd_head', 'd_inner', 'd_embed', 'div_val']
        for n in names:
          values = get_yaml_values(param_values[n][c])
          try:
            amlt_config['search']['params'].append({'name':n, 'spec':'discrete', 'values': [values]})
          except:
            amlt_config['search']['params'] = [{'name':n, 'spec':'discrete', 'values': [values]}]
        '''
        # run bundle_count jobs per sku
        amlt_config['environment']['image'] = 'debadeepta/pytorch:1.7.0-cuda11.0-cudnn8-devel'

        curr_configs = {k:param_values[k][c:c+bundle_count] for k in param_values.keys()}
        del amlt_config['search']
        amlt_config['jobs'] = [{}]
        amlt_config['jobs'][0]['name'] = f'config_{yaml_idx}'
        amlt_config['jobs'][0]['sku'] = f'G{n_gpus}'
        amlt_config['jobs'][0]['command'] = ['set -e -o xtrace', 'pip install --user -e .']
        amlt_config['jobs'][0]['command'] += get_bundle_run_command(curr_configs, parallel=False, default_config=default_config)

        config_file = f'nv_train_{yaml_idx}.yaml'
        f_name = os.path.join(path_to_configs, config_file)
        with open(f_name, 'w') as file:
            yaml.dump(amlt_config, file)
        
        c += bundle_count
        yaml_idx += 1

      n_configs = yaml_idx
      fname = 'amlt_run' + ('_fear_stage1' if activate_fear else '')
      exp_name = 'homogeneous' if homogeneous_layers else 'heterogeneous'
      exp_name = ('fear_stage_1_' if activate_fear else '') + exp_name
      generate_bash_files(path_to_configs, f_name=fname, exp_name=exp_name)

      local_test_bashfile = os.path.join(path_to_configs, 'local_test.sh')
      if os.path.exists(local_test_bashfile):
            os.remove(local_test_bashfile)  
      with open(local_test_bashfile, 'a') as f:
        for c in range(n_configs):
          keys = ['n_layer', 'n_head', 'd_model', 'd_head', 'd_inner', 'd_embed', 'div_val']
          values = []
          for k in keys:
            values.append(get_yaml_values(param_values[k][c]))
          command = 'python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer {} --n_head {} --d_model {} --d_head {} --d_inner {} --d_embed {} --div_val {} \n'.format(*values)
          f.write(command) 
          f.write('if [ $? -ne 0 ]; then \n echo FAIL \n exit \n fi \n')

  elif phase==2:
    bash_idx = 0
    while True:
      bash_file = os.path.join(path_to_configs, 'amlt_run_fear_stage2_'+str(bash_idx)+'.sh')
      if os.path.exists(bash_file):
            os.remove(bash_file)
          
      with open(bash_file, 'a') as f:
        for t in range(len(targets)):
          for i in range(batch_configs):
            job_idx = i + t * batch_configs + bash_idx * len(targets) * batch_configs
            print(job_idx)
            if job_idx >= n_configs:
              break
            
            if job_idx % batch_configs == 0:
              f.write('amlt map --yes -t {} ~/Projects/archaiphilly/map.yaml :fear_stage2_unfreeze_{} fear_stage_1 :train_xl_config_{}'.format(targets[t], n_unfreeze, job_idx))
            else:
              f.write(' :train_xl_config_{}'.format(job_idx))
          f.write(' fear_stage_2\n')  
          if job_idx >= n_configs:
              break
      
      if job_idx >= n_configs:
              break

      bash_idx += 1
      
  elif phase==3:
    files = os.listdir(path_to_configs)
    ref_exp_name = 'fear_stage_1_heterogeneous' # name of the amlt experiment with full runs to use as the ground-truth configutation
    for f in files:
      if re.search('(nv_train_[0-9]+.yaml)', f):
        with open(os.path.join(path_to_configs, f), 'r') as file:
          prev_config = yaml.load(file)

        job_name = 'train_xl_config_' + f.replace('.yaml','').split('_')[-1]
        gt_config_path = os.path.join('/home/t-mojanj/logdir/nv_xformer_xl/prev_jobs/{}'.format(ref_exp_name), job_name)
        path_to_gt_config = recurse_dir(gt_config_path, filename='config.yaml', path_to_ref=None)
        if path_to_gt_config:
          with open(path_to_gt_config, 'r') as f2:
            gt_config = yaml.load(f2)

          config_dict = ['n_layer', 'n_head', 'd_model', 'd_head', 'd_inner', 'd_embed', 'div_val']
          for k in config_dict:
            for param in prev_config['search']['params']:
              if param['name'] == k:
                if mismatch(gt_config[k], param['values'][0]):
                  print('mismatch found in {} inside config {}'.format(k, path_to_gt_config))
                break
          print('{} checked'.format(job_name))
        else:
          print('##### job {} previously did not run'.format(job_name))
        
        config_num = f.replace('.yaml','').split('_')[-1]

        if different_seeds:
          prev_config['search']['job_template']['name'] = 'train_xl_config_%s_%d' % (config_num, max_step)
        else:
          prev_config['search']['job_template']['name'] = 'train_xl_config_%s_500-5000' % (config_num)
        prev_config['search']['max_trials'] = 8
        prev_config['search']['job_template']['command'] = ['set -e -o xtrace', 'bash scripts/apex_install.sh', 'pip install --user -e .']
        if different_seeds:
          prev_config['search']['job_template']['command'] += [get_run_command(max_step, config_num, s) for s in different_seeds]
        else:
          prev_config['search']['job_template']['command'] += [get_run_command(i, config_num) for i in range(500, 5000, 500)]
        
        for idx, param in enumerate(prev_config['search']['params']):  
          if param['name'] == 'max_step':
            del prev_config['search']['params'][idx]
        
        with open(os.path.join(path_to_configs, f), 'w') as file:
          yaml.dump(prev_config, file)

    exp_name_base = 'fear_baseline' + ('_heterogeneous' if 'heterogeneous' in ref_exp_name else '')
    if different_seeds:
      exp_name = exp_name_base + '_{}_step'.format(max_step)
    else:
      exp_name = exp_name_base + '_constLR'

    generate_bash_files(path_to_configs, f_name='amlt_run_fear_baseline', exp_name=exp_name)

  elif phase==4:
    n_layers = [5]
    div_vals = [4]
    # d_inners = [512, 700]
    homogeneous_layers = False

    target_ppl_range = [40, 50]
    
    start_config = 63
    n_gpus = 8
    parallel_run = False
    gpu_config = 'dgx1_8gpu_fp32'
    targets = ['NLX-NDv2']

    '''proxy for n_params:
        Attn: 4 * d_model * d_model
        FFN: 2 * d_model * d_inner
    '''

    # # generate a random architecture
    # if homogeneous_layers:
    #   base_config = generate_params_homogeneous(n_configs=1)
    #   # pprint.pprint(new_param_values)
    # else:
    #   # base_config = generate_params_heterogeneous(n_configs=1)
    #   print('here')
    #   base_config = collections.OrderedDict([('n_layer', [6]), ('n_head', [[2,4,2,2,4,4]]), ('d_model', [128]), ('d_embed', [128]), ('div_val', [4]), ('d_head', [[64,32,64,64,32,32]]), ('d_inner', [[1229,1618,1901,952,1496,987]])])

    # find candidate architectures from previously seen configs
    exp_name = 'fear_stage_1_heterogeneous'
    path_to_results = os.path.join('/home/t-mojanj/logdir/nv_xformer_xl/prev_jobs/', exp_name)
    model_configs = gather_results(exp_name, path_to_results, filetypes='config.yaml')
    val_ppls = gather_results(exp_name, path_to_results, filetypes='.json')

    useful_configs = {}
    n_all_params = {}
    for config_name, model_config in model_configs.items():
      curr_val_ppl = val_ppls[config_name]['valid_perplexity']
      if curr_val_ppl >= target_ppl_range[0] and curr_val_ppl <= target_ppl_range[-1]:
        # model_config['attn_type'] = 0
        model_config['div_val'] = 4
        useful_configs[config_name] = model_config

        curr_n_all_param, params_adaptive_embedding, params_adaptive_softmax, params_attention, params_ff = get_model_and_params(model_config)
        n_all_params[config_name] = {'AdaEmb': float(params_adaptive_embedding), 'Sftmax': float(params_adaptive_softmax), \
                          'Attn': float(params_attention), 'FFN': float(params_ff), 'total': float(curr_n_all_param)}

        print(config_name, curr_val_ppl, n_all_params[config_name]['FFN']+n_all_params[config_name]['Attn'])

    # # which scaling factors are architecture dependant?
    # factors = []
    # for config_name in useful_configs.keys():
    #   model_config = useful_configs[config_name]
    #   model_config['d_inner'] = multiply(model_config['d_inner'], factor=2) + multiply(model_config['d_model'], factor=2.5)
    #   curr_n_all_param, params_adaptive_embedding, params_adaptive_softmax, params_attention, params_ff = get_model_and_params(model_config)
    #   new_params = params_attention + params_ff
    #   old_params =  n_all_params[config_name]['FFN'] + n_all_params[config_name]['Attn']
    #   print(config_name, new_params*1./old_params)
    #   factors.append(new_params*1./old_params)
    # print(len(factors), min(factors), max(factors))
    
    factors = np.linspace(3.0, 4.5, num=6).tolist()
    selected_configs = ['config_151']#np.random.choice(list(useful_configs.keys()), size=5).tolist()
    # selected_configs.remove('config_151')
    # knobs = ['n_layer', 'd_model', 'd_inner']
    knobs = ['d_inner']
    
    for config_name in selected_configs:
      #-------------- change from an intermediate config
      print(config_name, useful_configs[config_name]['d_model'])
      useful_configs[config_name]['d_model'] = 346 #multiply(useful_configs[config_name]['d_model'], 1.8)
      # factors = (np.asarray(factors)/3.).tolist()
    
    # find out architecture dependant scaling factors
    arch_factors = {}
    for config_name in selected_configs:
      arch_factors[config_name] = {k:{} for k in knobs}
      for k in knobs:
        for f in factors:
          if k=='n_layer':
            arch_factors[config_name][k][f] = 1.
          else:
            model_config = copy.deepcopy(useful_configs[config_name])
            model_config[k] = multiply(model_config[k], factor=f)
            if k=='d_model':
              model_config[k] += (model_config[k] % 2)
            elif k=='d_inner': 
              model_config[k] += multiply(useful_configs[config_name]['d_model'], factor=f)
            curr_n_all_param, params_adaptive_embedding, params_adaptive_softmax, params_attention, params_ff = get_model_and_params(model_config)
            new_params = params_attention + params_ff
            old_params =  n_all_params[config_name]['FFN'] + n_all_params[config_name]['Attn']
            arch_factors[config_name][k][f] = (old_params * f)/new_params
            
            step = 0.1 * arch_factors[config_name][k][f]
            iter = 0
            while iter<10:
              iter += 1
              model_config[k] = copy.deepcopy(useful_configs[config_name][k])
              model_config[k] = multiply(model_config[k], factor=f*arch_factors[config_name][k][f])
              if k=='d_model':
                model_config[k] += (model_config[k] % 2)
              elif k=='d_inner': 
                model_config[k] += multiply(useful_configs[config_name]['d_model'], factor=f*arch_factors[config_name][k][f])
              
              curr_n_all_param, params_adaptive_embedding, params_adaptive_softmax, params_attention, params_ff = get_model_and_params(model_config)
              new_params = params_attention + params_ff
              print('target:', f*old_params, 'achieved:', new_params)  
              if np.absolute(new_params - f*old_params)/old_params > 0.005:
                if new_params/old_params > f:
                  arch_factors[config_name][k][f] -= step
                  step /=2
                else:
                  arch_factors[config_name][k][f] += step
                  step /=2
              else:
                break 
      
      print('=>', config_name, arch_factors[config_name])
    
    param_keys = ['n_layer', 'n_head', 'd_model', 'd_embed', 'div_val', 'd_head', 'd_inner']
    bash_start = 0
    yaml_idx = start_config
    for config_name in selected_configs:
      base_config = useful_configs[config_name]
      print('base config:', base_config)
      param_values = {k: [] for k in param_keys}
      
      for f in factors:
        # generate architectures with scaled parameters
        for knob in knobs:
          for k in param_values.keys():
            if k==knob:
              if k=='n_layer':
                param_values[k].append(multiply(base_config[k], factor=f*arch_factors[config_name][k][f]))
                print('n_layer becomes', param_values[k][-1])
              elif k=='d_model':
                scaled_d_model = multiply(base_config[k], factor=f*arch_factors[config_name][k][f])
                scaled_d_model += (scaled_d_model % 2)
                param_values[k].append(scaled_d_model)
              else:
                param_values[k].append((multiply(base_config[k], factor=f*arch_factors[config_name][k][f])+multiply(base_config['d_model'], factor=f*arch_factors[config_name][k][f])).tolist())
            else:
              param_values[k].append(base_config[k])
      
      for i in range(len(param_values['n_layer'])):
        if param_values['n_layer'][i] > base_config['n_layer']:
          print(param_values['n_layer'][i], base_config['n_layer'])
          for k in param_values.keys():
            if isinstance(param_values[k][i], list):
              count = int(param_values['n_layer'][i]/base_config['n_layer'])
              res = param_values['n_layer'][i] % base_config['n_layer']
              param_values[k][i] = param_values[k][i] * count + param_values[k][i][:res]
      print(param_values)

      model_config = base_config
      for i in range(len(param_values['n_layer'])):
        for k in param_keys:
          model_config[k] = param_values[k][i]
        curr_n_all_param, params_adaptive_embedding, params_adaptive_softmax, params_attention, params_ff = get_model_and_params(model_config)
        new_params = params_attention + params_ff
        old_params =  n_all_params[config_name]['FFN']+n_all_params[config_name]['Attn']
        print(config_name, i, old_params, new_params, new_params*1./old_params)

      n_configs = len(param_values['n_layer'])
      # create corresponding yaml files for amulet jobs
      c = 0
      print('######', n_configs)
      while c < n_configs: 
        with open('/home/t-mojanj/Projects/archaiphilly/nv_train.yaml') as file:
          amlt_config = yaml.load(file)
          # if c==0:
          #   pprint.pprint(amlt_config)

        bundle_count = 8//n_gpus if parallel_run else 2
        curr_configs = {k:param_values[k][c:c+bundle_count] for k in param_values.keys()}

        amlt_config['environment']['setup'] = ['set -e -o xtrace', 'pip install --user tensorboard']
        amlt_config['environment']['image'] = 'debadeepta/pytorch:1.7.0-cuda11.0-cudnn8-devel'
        
        del amlt_config['search']
        amlt_config['jobs'] = [{}]
        amlt_config['jobs'][0]['name'] = '{}_{}'.format(config_name, str(c//bundle_count+start_config))
        amlt_config['jobs'][0]['sku'] = 'G8'
        amlt_config['jobs'][0]['command'] = ['set -e -o xtrace', 'pip install --user -e .']
        if parallel_run:
          amlt_config['jobs'][0]['command'].append(get_bundle_run_command(curr_configs, parallel=parallel_run))
        else:
          amlt_config['jobs'][0]['command'] += get_bundle_run_command(curr_configs, parallel=parallel_run)

        config_file = 'nv_train_'+str(int(yaml_idx))+'.yaml'
        f_name = os.path.join(path_to_configs, config_file)
        with open(f_name, 'w') as file:
            yaml.dump(amlt_config, file)

        c += bundle_count
        yaml_idx += 1

      n_configs = (yaml_idx-bash_start_config)
      print('######', n_configs)
      bash_start = generate_bash_files(path_to_configs, f_name='amlt_run_fear_stage1_similar_params', bash_start=bash_start, exp_name='simp')
      bash_start_config += n_configs

  else:
    n_configs = 16
    n_gpus = 8
    gpu_config = 'dgx1_8gpu_fp32'
    bundle_count = 6
    target = 'itpscusv100cl'

    dataset = 'lm1b'

    if dataset=='wt103':
      c = 0
      config_idx = 0
      while c < n_configs: 
          with open('../archaiphilly/nv_train.yaml') as file:
              amlt_config = yaml.safe_load(file)
              # if c==0:
              #   pprint.pprint(amlt_config)

          amlt_config['environment']['setup'] = ['set -e -o xtrace', 'pip install --user tensorboard']
          amlt_config['environment']['image'] = 'debadeepta/pytorch:1.7.0-cuda11.0-cudnn8-devel'
          
          del amlt_config['search']
          amlt_config['jobs'] = [{}]
          amlt_config['jobs'][0]['name'] = 'config_{}'.format(str(config_idx))
          amlt_config['jobs'][0]['sku'] = f'G{n_gpus}'
          amlt_config['jobs'][0]['command'] = ['set -e -o xtrace', 'pip install --user -e .']
          
          for i in range(1, bundle_count+1):
            n_layer = i + c
            exp_name = 'j' + str(i)

            if n_layer > n_configs:
              break
            
            amlt_config['jobs'][0]['command'].append('python -m torch.distributed.launch --nproc_per_node="%s" archai/nlp/nvidia_transformer_xl/train.py --config %s \
                          --config_file wt103_base.yaml --n_layer %s --n_head 8 --d_model 512 --d_head 64 --d_inner 2048 --div_val 4 --experiment_name %s' \
                          % (str(n_gpus), gpu_config, n_layer, exp_name))
          
          config_file = 'nv_train_'+str(config_idx)+'.yaml'
          path_to_configs = os.path.join('archai/nlp/nvidia_transformer_xl', 'configs')
          f_name = os.path.join(path_to_configs, config_file)
          with open(f_name, 'w') as file:
              yaml.dump(amlt_config, file)

          c += bundle_count
          config_idx += 1
    
    elif dataset=='lm1b':
      bundle_count = 10
      yaml_idx = 0
      c = 0
      while True:
        with open('../archaiphilly/transformer_nas/transxl_base_lm1b.yaml') as file:
          amlt_config = yaml.safe_load(file)
        del amlt_config['jobs'][0]['command'][-1]
        amlt_config['jobs'][0]['name'] = f'lm1b_config_{yaml_idx}'
        
        for i in range(1, bundle_count+1):
          n_layer = i + c
          exp_name = 'j' + str(i)

          if n_layer > n_configs:
            break
          
          command = 'python -m torch.distributed.launch --nproc_per_node={} archai/nlp/nvidia_transformer_xl/train.py \
                    --cuda --dataset lm1b --adaptive --div_val 4 --n_layer {} --d_model 512 --n_head 8 --d_head 64 \
                    --d_inner 2048 --dropout 0.00 --dropatt 0.00 --optim adam --warmup_step 20000 --max_step 100000 \
                    --lr 0.00025 --eta_min 0.0 --tgt_len 32 --mem_len 32 --eval_tgt_len 32 --batch_size 224 --multi_gpu ddp \
                    --gpu0_bsz 32 --experiment_name {}'.format(n_gpus, n_layer, exp_name)
          amlt_config['jobs'][0]['command'].append(command)
          
        config_file = 'nv_train_'+str(yaml_idx)+'.yaml'
        path_to_configs = '../archaiphilly/transformer_nas/lm1b_transformerXL_baseline'
        os.makedirs(path_to_configs, exist_ok=True)        
        f_name = os.path.join(path_to_configs, config_file)
        with open(f_name, 'w') as file:
            yaml.dump(amlt_config, file)
        
        if n_layer > n_configs:
          break

        c += bundle_count
        yaml_idx += 1

    # exp_name = f'memformer_baselines_{dataset}'
    # bash_f_name = 'amlt_run_memformerBase.sh'
    # bash_file = os.path.join(path_to_configs, bash_f_name)
    # if os.path.exists(bash_file):
    #     os.remove(bash_file)  
    # for i in range(config_idx):
    #     with open(bash_file, 'a') as f:
    #         f.write('amlt run --yes archai/nlp/nvidia_transformer_xl/configs/nv_train_{}.yaml {} -t {}\n'.format(i, exp_name, target))
