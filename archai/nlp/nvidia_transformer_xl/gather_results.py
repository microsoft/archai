import os
from pickle import TRUE
import numpy as np
import collections
import yaml
import collections
import argparse
import json
import re
import pprint
import types
import pandas as pd
from functools import partial
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import torch
import torch.nn as nn
import torch.nn.functional as F

from archai.common import utils, common
from archai.nlp.nvidia_transformer_xl.data_utils import get_lm_corpus
from archai.nlp.nvidia_transformer_xl.nvidia_utils import exp_utils
from archai.nlp.nvidia_transformer_xl.mem_transformer import MemTransformerLM #, MemTransformerLM_flex
from archai.nlp.nvidia_transformer_xl.mem_transformer import PositionwiseFF, MultiHeadAttn, RelMultiHeadAttn, \
                                                            RelPartialLearnableMultiHeadAttn, RelLearnableMultiHeadAttn, DecoderLayer, \
                                                            RelLearnableDecoderLayer, RelPartialLearnableDecoderLayer, AdaptiveEmbedding, ProjectedAdaptiveLogSoftmax
from archai.nlp.nvidia_transformer_xl.nvidia_utils.log_uniform_sampler import sample_logits
from archai.nlp.nvidia_transformer_xl.utils import process_parameters
from archai.nlp.nvidia_transformer_xl.profiler import get_model_profile

def meta_constructor_mapping(loader, node):
    value = loader.construct_mapping(node)
    return value

def meta_constructor_sequence(loader, node):
    value = loader.construct_sequence(node)
    return value

yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar', meta_constructor_sequence)
yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.dtype', meta_constructor_mapping)

model_config_keys = ['n_token', 'n_layer','n_head','d_model','d_head','d_inner','dropout','dropatt', \
                        'd_embed','div_val','pre_lnorm','tgt_len','ext_len','mem_len', \
                        'same_length','attn_type','clamp_len','sample_softmax']


def get_label(baseline_config, new_config):
  label = []
  for k in ['n_layer','d_model','d_inner']:
    if np.any(baseline_config[k] != new_config[k]):
      label.append(k)

  if 'n_layer' in label:
    label.remove('d_inner')

  return '_'.join(label)


def forward_with_output_memtransformer(self, data, target, mems):
    # nn.DataParallel does not allow size(0) tensors to be broadcasted.
    # So, have to initialize size(0) mems inside the model forward.
    # Moreover, have to return new_mems to allow nn.DataParallel to piece
    # them together.
    if mems is None:
        mems = self.init_mems()

    tgt_len = target.size(0)
    hidden, new_mems = self._forward(data, mems=mems)

    pred_hid = hidden[-tgt_len:]
    logits = None
    if self.sample_softmax > 0 and self.training:
        assert self.tie_weight
        logit = sample_logits(self.word_emb, self.out_layer.bias, target,
                                pred_hid, self.sampler)
        loss = -F.log_softmax(logit, -1)[:, :, 0]
    else:
        out, loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))


def get_metrics(topk, sorted_ground_truth, sorted_target, val_ppl_list_gt, val_ppl_list_target, common_configs=None):
  idx = int(topk/100.*len(sorted_ground_truth))
  sorted_ground_truth_binned = sorted_ground_truth[:idx].astype(np.int32)
  sorted_target_binned = sorted_target[:idx].astype(np.int32)

  correct = len(np.intersect1d(sorted_target_binned, sorted_ground_truth_binned))
  total = len(sorted_target_binned)
  common_ratio = correct*1./total
  print('Correctly ranked top %d %% (%d) with %.2f accuracy'%(topk, total, correct*1./total))

  topk_val_ppl_list_gt = [val_ppl_list_gt[i] for i in range(len(val_ppl_list_gt)) if i in sorted_ground_truth_binned]
  topk_val_ppl_list_target = [val_ppl_list_target[i] for i in range(len(val_ppl_list_target)) if i in sorted_ground_truth_binned]
  # topk_val_ppl_list_gt = [val_ppl_list_gt[i] for i in range(len(val_ppl_list_gt)) if i in sorted_target_binned]
  # topk_val_ppl_list_target = [val_ppl_list_target[i] for i in range(len(val_ppl_list_target)) if i in sorted_target_binned]
  spr_rank, _ = spearmanr(topk_val_ppl_list_gt, topk_val_ppl_list_target)
  print('Spearman Correlation on top %d %% (%d): %.3f'%(topk, len(topk_val_ppl_list_gt), spr_rank))

  return common_ratio, spr_rank
                          

def extract_date_time(log_str):
  idx_end_time = list(re.search(' - INFO - ', log_str).span())[0]
  idx_start_time = idx_end_time-8
  time_str = log_str[idx_start_time:idx_end_time]

  date_str = log_str[:idx_start_time-1]
  y, mo, d = date_str.split('-')

  h, m, s = time_str.split(':')

  return int(y), int(mo), int(d), int(h), int(m), int(s)


def extract_step(log_str):
  step = log_str.split('|')[1].split('step')[-1]
  return step


def get_info_from_logs(log_file, stage_1=True):
  out_dict = {}
  
  with open(log_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
  
  ppl_thresholds = None
  for idx, l in enumerate(lines):
    if 'ppl_threshold' in l:
      if ppl_thresholds is None:
        try:
          str = re.search('\[(.+?)\]', l).group(1)
          ppl_thresholds = [int(float(thr)) for thr in str.split(',')]
        except:
          str = re.search('ppl_threshold=([0-9]+),', l).group(1)
          ppl_thresholds = int(float(str))
    
    elif '#params' in l:
      n_params = re.search('#params = ([0-9]+)', l).group(1)
      out_dict['n_params'] = n_params
    
    elif '#non emb params' in l:
      start_y, start_mo, start_d, start_h, start_m, start_s = extract_date_time(lines[idx+1])
    
    elif 'Saving FEAR checkpoint' in l:
      end_y, end_mo, end_d, end_h, end_m, end_s = extract_date_time(l)
      idx_ppl = re.search('ppl', l).span()[-1]
      ppl = float(l.replace('\n','')[idx_ppl:])
      ppl_key = min(ppl_thresholds, key=lambda x:abs(x-int(ppl)))

      out_dict[ppl_key] = {}
      out_dict[ppl_key]['ppl'] = ppl
      out_dict[ppl_key]['time'] = (end_y-start_y)*365*24*3600 + (end_mo-start_mo)*30*24*3600 + (end_d-start_d)*24*3600 + (end_h-start_h)*3600 + (end_m-start_m)*60 + (end_s-start_s)
      assert out_dict[ppl_key]['time'] > 0, print(end_y, end_mo, end_d, end_h, end_m, end_s, start_y, start_mo, start_d, start_h, start_m, start_s)
      step = int(extract_step(lines[idx-2]))
      out_dict[ppl_key]['step'] = step

    elif 'Training time:' in l and stage_1:
      t_minutes = re.search('Training time: ([+-]?([0-9]*[.])?[0-9]+) minutes', l).group(1)
      out_dict['train_time'] = float(t_minutes) * 60
  
  return out_dict


def get_info_from_json(json_file, step=[], type=None):
  '''
    step: step number to extract the ppl log, live empty to get the final ppl 
    type: select from [test, valid, train]
  '''
  out_dict = {}
  key = type+'_perplexity' if type is not None else None
  
  with open(json_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()[::-1]
    try:
      job_desc = re.search('DLLL \{(.+?)\}', lines[-1])
    except:
      return None
    job_desc = '{'+job_desc.group(1)+'}}'
    work_dir = json.loads(job_desc)['data']['work_dir']
    idx_start = re.search('amlt-results', work_dir).span()[-1] + 1
    amlt_job = work_dir[idx_start:].split('/')[0]
  
    for line in lines:
      str = re.search('DLLL \{(.+?)\}', line)
      str = '{'+str.group(1)+'}}'
      final_train_log = json.loads(str)

      if len(step)>0:
        if final_train_log['step']==[]:
          out_dict['train_elapsed'] = float(final_train_log['data']['train_elapsed'])*60
          for k in final_train_log['data'].keys():
            if 'perplexity' in k:
              out_dict[k] = final_train_log['data'][k]
        
        elif final_train_log['step'][0] in step:
          if final_train_log['step'][0] not in out_dict.keys():
            out_dict[final_train_log['step'][0]] = {}
          for k in final_train_log['data'].keys():
            if 'perplexity' in k:
              out_dict[final_train_log['step'][0]][k] = final_train_log['data'][k]
          # if key is None:
          #   out_dict[final_train_log['step'][0]] = {}
          #   for k in final_train_log['data'].keys():
          #     if 'perplexity' in k:
          #       out_dict[final_train_log['step'][0]][k] = final_train_log['data'][k]
          # elif key in final_train_log['data'].keys():
          #   out_dict[key] = final_train_log['data'][key]

      else:
        try:
          out_dict['train_elapsed'] = float(final_train_log['data']['train_elapsed'])*60
          if key is None:
            for k in final_train_log['data'].keys():
              if 'perplexity' in k:
                out_dict[k] = final_train_log['data'][k]
          elif key in final_train_log['data'].keys():
            out_dict[key] = final_train_log['data'][key]
          out_dict['amlt_job'] = amlt_job
          break
        except:
          return None
  
  return out_dict


def get_config_name(job):
  # idx = list(re.search('config_', job).span())[0]
  # return job[idx:]
  if 'baseline' in job:
    dir_name = os.path.basename(os.path.dirname(job))
    return re.search('(config_[0-9]+_[0-9]+)', dir_name).group(1)
  elif 'similar_params_sweep' in job or 'simp' in job or 'evolution' in job:
    idx =  re.search('(config_[0-9]+)', job).span()[0]
    job = job[idx:]
    config_name = job.split('/')[0]
    return config_name + '_' + job.split('/')[1]
  elif 'evo_search' in job or 'midevolution' in job:
    return job
  else:
    return re.search('(config_[0-9]+)', job).group(1)
    

def recurse_dir(args, exp_name, path_to_dir, filetypes='.json'):
  if not isinstance(filetypes, list):
    filetypes = [filetypes]
  
  results = {}
  for j in os.listdir(path_to_dir):
      j_path = os.path.join(path_to_dir, j)
      if os.path.isdir(j_path):
        if args.n_unfreeze is not None and 'unfreeze_{}'.format(args.n_unfreeze) not in j:
          continue
        results.update(recurse_dir(args, exp_name, j_path, filetypes))
      else:
        for ft in filetypes:
          if ft in j:
            if ft=='.log':
              logs = get_info_from_logs(j_path, stage_1='stage_1' in exp_name)
            elif ft=='.json':
              logs = get_info_from_json(j_path, step=args.step, type=args.log_type)
            elif ft=='config.yaml':
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
              print(config_name, logs)
              if config_name in results.keys():
                results[config_name].update(logs)
              else:
                results[config_name] = logs
  
  return results


# def recurse_dir(args, exp_name, path_to_dir, read_log_file=False):
#   results = {}
#   for j in os.listdir(path_to_dir):
#       j_path = os.path.join(path_to_dir, j)
#       if os.path.isdir(j_path):
#         if args.n_unfreeze is not None and 'unfreeze_{}'.format(args.n_unfreeze) not in j:
#           continue
#         results.update(recurse_dir(args, exp_name, j_path, read_log_file))
#       else:
#         if '.log' in j and read_log_file:
#           logs_log = get_info_from_logs(j_path, stage_1='stage_1' in exp_name)
#           if logs_log: 
#             config_name = get_config_name(j_path) #get_config_name(os.path.basename(os.path.dirname(j_path)))
#             print(config_name, logs_log)
#             if config_name in results.keys():
#               results[config_name].update(logs_log)
#             else:
#               results[config_name] = logs_log

#         if '.json' in j:
#           json_logs = get_info_from_json(j_path, step=args.step, type=args.log_type)
#           if json_logs: 
#             config_name = get_config_name(j_path) #get_config_name(os.path.basename(os.path.dirname(j_path)))
#             print(config_name, json_logs)
#             if config_name in results.keys():
#               results[config_name].update(json_logs)
#             else:
#               results[config_name] = json_logs
  
#   return results


def get_parser():
  parser = argparse.ArgumentParser(description='Results Analysis.')
  parser.add_argument('--results_dir', type=str, default='/home/t-mojanj/logdir/nv_xformer_xl/prev_jobs/',
                      help='path where amulet results are downloaded')
  parser.add_argument('--exp_name', type=lambda s: [item for item in s.split(',')], #required=True,
                      help='name of maulet experiment')
  parser.add_argument('--step', type=lambda s: [int(item) for item in s.split(',')], default=[],
                      help='training step to extract the log from')
  parser.add_argument('--cross_step', action='store_true',
                      help='analyze metrics across different steps of fear stage 2')     
  parser.add_argument('--log_type', type=str, default=None,
                      help='type of ppl log to extract, select from [test, valid, train]')
  parser.add_argument('--n_unfreeze', type=int, default=None,
                      help='number of unforzen layers for fear_stage_2')
  parser.add_argument('--analyze', action='store_true',
                      help='analyze yaml results and generate metrics versus topk')
  parser.add_argument('--read_jsons', action='store_true',
                      help='read json results and summarize in a yaml file')
  parser.add_argument('--generate_plots', action='store_true',
                      help='generate spearman correlation and common ratio plots with baseline included')
  parser.add_argument('--analyze_params', action='store_true',
                      help='analyze model parameter size')     
  parser.add_argument('--param_ranking', action='store_true',
                      help='generate metrics w.r.t parameter size')    
  parser.add_argument('--cross_seeds', action='store_true',
                      help='generate metrics across various seeds')        
  parser.add_argument('--animation', action='store_true',
                      help='create animation of models training')   
  parser.add_argument('--export_to_csv', action='store_true',
                      help='export model configs and ppls to csv')   
  parser.add_argument('--similar_params', action='store_true',
                      help='plot ppl versus parameter size for similar param experiment')          

  args = parser.parse_args()
  return args


def main(args):
  if args.analyze:    # provides 2 plots: 1) common ratio versus topk and 2) spearman corr versus topk
    results = {}
    common_ratios = {}
    spr_ranks = {}
    
    fname = 'result_summary.yaml'
    yaml_file = os.path.join(os.path.join(args.results_dir, 'fear_stage_1'), fname)
    assert os.path.exists(yaml_file), 'no result summary for the ground-truth job'
    with open(yaml_file, 'r') as f:
      results['fear_stage_1'] = collections.OrderedDict(yaml.safe_load(f))

    for n_unfreeze in [3]:#[2,3]:
      for i, exp_name in enumerate(args.exp_name):
        path_to_results = os.path.join(args.results_dir, exp_name)
        assert 'stage_2' in exp_name
        fname = 'result_summary_unfreeze_{}.yaml'.format(n_unfreeze) #'2' if args.n_unfreeze is None else args.n_unfreeze)
        ppl_threshold = 70 if 'ppl' not in exp_name else re.search('ppl_([0-9]+)', exp_name).group(1)
        exp_name = 'fear_stage_2'

        yaml_file = os.path.join(path_to_results, fname)
        if not os.path.exists(yaml_file):
          print('#### no yaml summary found for {} with n_unfreeze={}'.format(args.exp_name[i], n_unfreeze))
          continue
        
        with open(yaml_file, 'r') as f:
          results[exp_name] = collections.OrderedDict(yaml.safe_load(f))

        common_configs = np.intersect1d(list(results['fear_stage_1'].keys()), list(results['fear_stage_2'].keys()))
        print('analyzing {} architectures'.format(len(common_configs)))
        
        # fear_stage_1 results:
        val_ppl_list_stage1 = []
        for k in common_configs:
          val_ppl_list_stage1.append(results['fear_stage_1'][k]['valid_perplexity'])
        sorted_ground_truth = np.argsort(val_ppl_list_stage1)

        # fear_stage_2 results:
        val_ppl_list_stage2 = []
        for k in common_configs:
          val_ppl_list_stage2.append(results['fear_stage_2'][k]['valid_perplexity'])

        sorted_fear = np.argsort(val_ppl_list_stage2)
        
        # extract common ratio and spearmanrank
        key = 'n_unfreeze_{}_ppl_{}'.format(n_unfreeze, ppl_threshold)
        print('--------------- ', key)
        common_ratios[key] = []
        spr_ranks[key] = []

        topk_list = range(10,101,10)
        for topk in topk_list:
          common_ratio, spr_rank = get_metrics(topk, sorted_ground_truth, sorted_target=sorted_fear, \
                                                val_ppl_list_gt=val_ppl_list_stage1, val_ppl_list_target=val_ppl_list_stage2)
          common_ratios[key].append(common_ratio)
          spr_ranks[key].append(spr_rank)

    plt.figure()
    for k in common_ratios.keys():
      plt.plot(topk_list, common_ratios[k], label=k, marker='.', markersize=10)
    plt.ylabel('Common ratio')
    plt.xlabel('Topk (%)')
    plt.xticks(topk_list)
    plt.legend(loc='lower right')
    plt.grid(axis='y')
    plt.savefig('common_ratio_topk.png', bbox_inches="tight")

    plt.figure()
    for k in spr_ranks.keys():
      plt.plot(topk_list, spr_ranks[k], label=k, marker='.', markersize=10)
      # for i, label in enumerate(spr_ranks[k]):
      #   plt.text(topk_list[i], spr_ranks[k][i]-0.07, '%.2f'%label)

    plt.ylabel('Spearman\'s Correlation')
    plt.xlabel('Topk (%)')
    plt.xticks(topk_list)
    plt.ylim(top=1)
    plt.grid(axis='y')
    plt.legend(loc='lower right')
    plt.savefig('spearman_topk.png', bbox_inches="tight")


  elif args.cross_seeds: # provides the plots similar to args.analyze but across different random seeds of the same experiment
    results = {}
    common_ratios = {}
    spr_ranks = {}
    
    fname = 'result_summary.yaml'
    yaml_file = os.path.join(os.path.join(args.results_dir, 'fear_stage_1'), fname)
    assert os.path.exists(yaml_file), 'no result summary for the ground-truth job'
    with open(yaml_file, 'r') as f:
      results['fear_stage_1'] = collections.OrderedDict(yaml.safe_load(f))

    for n_unfreeze in [2,3]:
      for i, exp_name in enumerate(args.exp_name):
        path_to_results = os.path.join(args.results_dir, exp_name)
        if 'stage_2' in exp_name:
          fname = 'result_summary_unfreeze_{}.yaml'.format(n_unfreeze) #'2' if args.n_unfreeze is None else args.n_unfreeze)
          ppl_threshold = 70 if 'ppl' not in exp_name else re.search('ppl_([0-9]+)', exp_name).group(1)
          exp_name = 'fear_stage_2'
        else:
          fname = 'result_summary.yaml'

        yaml_file = os.path.join(path_to_results, fname)
        if not os.path.exists(yaml_file):
          print('#### no yaml summary found for {} with n_unfreeze={}'.format(args.exp_name[i], n_unfreeze))
          continue
        with open(yaml_file, 'r') as f:
          results[exp_name] = collections.OrderedDict(yaml.safe_load(f))

        structured_results = {}
        for k in results[exp_name].keys():
          config_name = re.search('(config_[0-9]+)', k).group(1)
          seed = re.search('(seed_[0-9]+)', k).group(1)
          try:
            structured_results[seed][config_name] = results[exp_name][k]
          except:
            structured_results[seed] = {}
            structured_results[seed][config_name] = results[exp_name][k]

        # gather results:
        common_configs = {}
        val_ppl_list_target = {}
        val_ppl_list_gt = {}
        
        for seed in structured_results.keys():
          common_configs[seed] = np.intersect1d(list(results['fear_stage_1'].keys()), list(structured_results[seed].keys()))
          print('{} has {} configs'.format(seed, len(common_configs[seed])))
          
          val_ppl_list_target[seed] = []
          val_ppl_list_gt[seed] = []
          for conf in common_configs[seed]:
            val_ppl_list_target[seed].append(structured_results[seed][conf]['valid_perplexity'])
            val_ppl_list_gt[seed].append(results['fear_stage_1'][conf]['valid_perplexity'])
          
        
        sorted_target = {}
        for seed in val_ppl_list_target.keys():
          sorted_target[seed] = np.argsort(val_ppl_list_target[seed])
        
        # extract common ratio and spearmanrank
        topk_list = range(10,101,10)
        for seed in val_ppl_list_target.keys():
          print('--------------- ', seed)
          sorted_ground_truth = np.argsort(val_ppl_list_gt[seed])
          sorted_target = np.argsort(val_ppl_list_target[seed])

          common_ratios[seed] = []
          spr_ranks[seed] = []
          for topk in topk_list:
            common_ratio, spr_rank = get_metrics(topk, sorted_ground_truth, sorted_target=sorted_target, \
                                                  val_ppl_list_gt=val_ppl_list_gt[seed], val_ppl_list_target=val_ppl_list_target[seed])
            common_ratios[seed].append(common_ratio)
            spr_ranks[seed].append(spr_rank)

    # for i, topk in enumerate(topk_list):
    #   plt.figure()
    #   for k in common_ratios.keys():
    #     if k=='seed_1111':
    #         plt.scatter(150, common_ratios[k][i], marker='.', s=150, c='midnightblue', label='Common Ratio')
    #         plt.scatter(150, spr_ranks[k][i], marker='^', s=100, c='tab:blue', label='Spearman Correlation')
    #     else:
    #       plt.scatter(150, common_ratios[k][i], marker='.', s=150, c='midnightblue')
    #       plt.scatter(150, spr_ranks[k][i], marker='^', s=100, c='tab:blue')
    #   plt.ylabel('Metric')
    #   plt.xlabel('Time (s)')
    #   plt.title('Topk = %d %%' % topk)
    #   plt.grid(axis='y')
    #   plt.xlim((0, 3000))
    #   plt.ylim((0.0,1.1))
    #   plt.legend(loc='lower right')#'center left', bbox_to_anchor=(1, 0.5))
    #   plt.savefig('common_ratio_spearman_topk_{}.png'.format(topk), bbox_inches="tight")
    
    plt.figure()
    for k in common_ratios.keys():
      plt.plot(topk_list, common_ratios[k], label=k, marker='.', markersize=10)
    plt.ylabel('Common ratio')
    plt.xlabel('Topk (%)')
    plt.xticks(topk_list)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(axis='y')
    plt.savefig('common_ratio_topk_seeds.png', bbox_inches="tight")

    plt.figure()
    for k in spr_ranks.keys():
      plt.plot(topk_list, spr_ranks[k], label=k, marker='.', markersize=10)
      # for i, label in enumerate(spr_ranks[k]):
      #   plt.text(topk_list[i], spr_ranks[k][i]-0.07, '%.2f'%label)
    plt.ylabel('Spearman\'s Correlation')
    plt.xlabel('Topk (%)')
    plt.xticks(topk_list)
    plt.ylim(top=1)
    plt.grid(axis='y')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('spearman_topk_seeds.png', bbox_inches="tight")
    

  elif args.read_jsons: # extracts information from the .json training log (see get_info_from_json function above)
    for exp_name in args.exp_name:
      path_to_results = os.path.join(args.results_dir, exp_name)

      results = {}
      results = recurse_dir(args, exp_name, path_to_results, filetypes='.json')
        
      print('found %d configurations'%len(results.keys()))
      if 'stage_2' in exp_name:
        fname = 'result_summary_unfreeze_{}.yaml'.format('2' if args.n_unfreeze is None else args.n_unfreeze)
      else:
        fname = 'result_summary.yaml'
      yaml_file = os.path.join(path_to_results, fname)
      with open(yaml_file, 'w') as f:
        yaml.dump(results, f)
        print('saved results summary to', fname)


  elif args.generate_plots: # creates 2 plots per topk value: 1) common ratio versus training time and 2) spearman corr versus training time
    results = {}
    results_structured = {}
    common_ratios = {}
    spr_ranks = {}
    times = {}
    topk_list = [10,20,30,40,50,100]
    
    # load groundtruth results 
    for exp_name in args.exp_name:
      if 'fear_stage_1' in exp_name:
        ref_exp_name = exp_name
        print('reference experiment name is {}'.format(ref_exp_name))
        break
    path_to_results = os.path.join(args.results_dir, ref_exp_name)
    yaml_file = os.path.join(path_to_results, 'result_summary.yaml')
    with open(yaml_file, 'r') as f:
      results['fear_stage_1'] = collections.OrderedDict(yaml.safe_load(f))

    scores = None
    files = os.listdir(path_to_results)
    found_synflow = False
    for f in files:
      if False:#'synflow_scores' in f:
        found_synflow = True
        break
    yaml_file = os.path.join(path_to_results, f)
    if found_synflow:
      print('analyzing metrics for synflow')
      with open(yaml_file, 'r') as f:
        scores = yaml.safe_load(f)
      
      common_configs = np.intersect1d(list(results['fear_stage_1'].keys()), list(scores.keys()))
      print('analyzing {} architectures'.format(len(common_configs)))

      # fear_stage_1 results:
      val_ppl_list_gt = []
      for k in common_configs:
        val_ppl_list_gt.append(results['fear_stage_1'][k]['valid_perplexity'])
      sorted_ground_truth = np.argsort(val_ppl_list_gt)

      # zero-cost score results:
      target_scores = []
      for k in common_configs:
        target_scores.append(-scores[k])   # the higher the score, the better the architecture (reversely correlated with ppl)
      sorted_target = np.argsort(target_scores)

      common_ratios_synflow = {}
      spr_ranks_synflow = {}
      # extract common ratio and spearmanrank
      for topk in topk_list:
        common_ratio, spr_rank = get_metrics(topk, sorted_ground_truth=sorted_ground_truth, sorted_target=sorted_target, \
                                              val_ppl_list_gt=val_ppl_list_gt, val_ppl_list_target=target_scores)
        common_ratios_synflow[topk] = common_ratio
        spr_ranks_synflow[topk] = spr_rank

    for exp_name in args.exp_name:
      if 'stage_1' in exp_name:
        continue
      path_to_results = os.path.join(args.results_dir, exp_name)

      if 'stage_2' in exp_name:
        fname = 'result_summary_unfreeze_{}.yaml'.format(args.n_unfreeze) #'2' if args.n_unfreeze is None else args.n_unfreeze)
        target_ppl = 70 if 'ppl' not in exp_name else int(re.search('ppl_([0-9]+)', exp_name).group(1))
        legend_key = 'fear, ppl:{}'.format(target_ppl)
      else:
        fname = 'result_summary.yaml'
        legend_key = exp_name.replace('_', ' ')
        if 'baseline' in legend_key:
          legend_key = legend_key.replace('fear', '')

      yaml_file = os.path.join(path_to_results, fname)
      with open(yaml_file, 'r') as f:
        results[exp_name] = collections.OrderedDict(yaml.safe_load(f))

      common_ratios[legend_key] = {}
      spr_ranks[legend_key] = {}
      times[legend_key] = {}
      # common_ratios[legend_key+'_bottom'] = {}
      # spr_ranks[legend_key+'_bottom'] = {}
      # times[legend_key+'_bottom'] = {}

      if 'fear_baseline' in exp_name:
        for k, v in results[exp_name].items():
          max_step = k.split('_')[-1]
          config_name = re.search('(config_[0-9]+)_', k).group(1)
          if max_step not in results_structured.keys():
            results_structured[max_step] = {}
          results_structured[max_step][config_name] = v  

        # parse fear_baseline results:
        val_ppl_list_gt = {}
        val_ppl_list_baseline = {}
        timing_baseline = {}
        common_configs = {}
        
        if 'constLR' in exp_name:
          max_steps = str(500)#[str(i) for i in range(500, 5000, 500)] #[str(i) for i in range(5000, 40000, 5000)] + ['2500', '500']
        else:
          max_steps = str(5000) #[str(i) for i in range(5000, 40000, 5000)] #+ ['2500', '500']
        for max_step, v in results_structured.items():
          if not max_step in max_steps:
            continue
          val_ppl_list_baseline[max_step] = []
          timing_baseline[max_step] = []
          val_ppl_list_gt[max_step] = []
          
          common_configs[max_step] = np.intersect1d(list(results['fear_stage_1'].keys()), list(results_structured[max_step].keys()))
          for k in common_configs[max_step]:
            val_ppl_list_baseline[max_step].append(results_structured[max_step][k]['valid_perplexity'])
            timing_baseline[max_step].append(results_structured[max_step][k]['train_elapsed'])
            val_ppl_list_gt[max_step].append(results['fear_stage_1'][k]['valid_perplexity'])

        for topk in topk_list:
          common_ratios[legend_key][topk] = []
          spr_ranks[legend_key][topk] = []
          times[legend_key][topk] = []
          # common_ratios[legend_key+'_bottom'][topk] = []
          # spr_ranks[legend_key+'_bottom'][topk] = []
          # times[legend_key+'_bottom'][topk] = []
          for max_step in val_ppl_list_baseline.keys():
            # if int(max_step) >= 5000:
            #   continue
            print('------------ {} total number of configs with steps={}'.format(len(val_ppl_list_gt[max_step]), max_step))
            sorted_ground_truth = np.argsort(val_ppl_list_gt[max_step])
            sorted_baseline = np.argsort(val_ppl_list_baseline[max_step])
            common_ratio, spr_rank = get_metrics(topk, sorted_ground_truth=sorted_ground_truth, sorted_target=sorted_baseline, \
                                                    val_ppl_list_gt=val_ppl_list_gt[max_step], val_ppl_list_target=val_ppl_list_baseline[max_step], 
                                                    common_configs=common_configs[max_step])

            # common_ratio_bottom, spr_rank_bottom = get_metrics(topk, sorted_ground_truth=sorted_ground_truth[::-1], sorted_target=sorted_baseline[::-1], \
            #                                         val_ppl_list_gt=val_ppl_list_gt[max_step], val_ppl_list_target=val_ppl_list_baseline[max_step], 
            #                                         common_configs=common_configs[max_step])
            
            common_ratios[legend_key][topk].append(common_ratio)
            spr_ranks[legend_key][topk].append(spr_rank)
            times[legend_key][topk].append(np.average(timing_baseline[max_step]))

            # common_ratios[legend_key+'_bottom'][topk].append(common_ratio_bottom)
            # spr_ranks[legend_key+'_bottom'][topk].append(spr_rank_bottom)
            # times[legend_key+'_bottom'][topk].append(np.average(timing_baseline[max_step]))
      
      elif 'fear_stage_2' in exp_name:
        common_configs_stage2 = np.intersect1d(list(results['fear_stage_1'].keys()), list(results[exp_name].keys()))
        
        # parse fear_stage_1 results:
        val_ppl_list_gt_for_fear = []
        for k in common_configs_stage2:
          val_ppl_list_gt_for_fear.append(results['fear_stage_1'][k]['valid_perplexity'])
        sorted_ground_truth_for_fear = np.argsort(val_ppl_list_gt_for_fear)
        
        # parse fear_stage_2 results:
        val_ppl_list_stage2 = []
        timing_stage2 = []
        for k in common_configs_stage2:
          val_ppl_list_stage2.append(results[exp_name][k]['valid_perplexity'])
          timing_stage2.append(results[exp_name][k]['train_elapsed'] + results['fear_stage_1'][k][target_ppl]['time'])
        sorted_fear = np.argsort(val_ppl_list_stage2)
        
        # extract common ratio and spearmanrank
        for topk in topk_list:
          common_ratio_fear, spr_rank_fear = get_metrics(topk, sorted_ground_truth=sorted_ground_truth_for_fear, sorted_target=sorted_fear, \
                                                  val_ppl_list_gt=val_ppl_list_gt_for_fear, val_ppl_list_target=val_ppl_list_stage2, common_configs=common_configs_stage2)
              
          common_ratios[legend_key][topk] = common_ratio_fear
          spr_ranks[legend_key][topk] = spr_rank_fear
          times[legend_key][topk] = np.average(timing_stage2)

    markers = ['.', '^', '*', 'v', 'd', 'X', 's']
    for topk in topk_list:
      plt.figure()
      for i, k in enumerate(common_ratios.keys()):
        if 'fear' in k:
          plt.scatter(times[k][topk], common_ratios[k][topk], marker='.', s=150, c='limegreen')
          plt.scatter(times[k][topk], spr_ranks[k][topk], marker='^', s=100, c='limegreen')
        else:
          plt.scatter(times[k][topk], common_ratios[k][topk], label='Common Ratio', marker='.', s=150, c='midnightblue')
          plt.scatter(times[k][topk], spr_ranks[k][topk], label='Spearman Correlation', marker='^', s=100)
      if scores:
        plt.scatter(0, common_ratios_synflow[topk], label='synflow', marker=markers[i+1], s=80)
      plt.ylabel('Metric')
      plt.xlabel('Time (s)')
      plt.title('Topk = %d %%' % topk)
      plt.grid(axis='y')
      plt.xlim((0, 3000))
      plt.ylim((0.0,1.1))
      # plt.legend(loc='lower right')#'center left', bbox_to_anchor=(1, 0.5))
      plt.savefig('common_ratio_spearman_topk_{}.png'.format(topk), bbox_inches="tight")
    
    
    markers = ['.', '*', 'v', 'd', 'X', 's']
    for topk in topk_list:
      plt.figure()
      for i, k in enumerate(common_ratios.keys()):
        plt.scatter(times[k][topk], common_ratios[k][topk], label=k, marker=markers[i], s=150)
      if scores:
        plt.scatter(0, common_ratios_synflow[topk], label='synflow', marker=markers[i+1], s=80)
      plt.ylabel('Common ratio')
      plt.xlabel('Time (s)')
      plt.title('Topk = %d %%' % topk)
      plt.grid(axis='y')
      plt.ylim((0.2,1.1))
      # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
      plt.savefig('common_ratio_topk_{}.png'.format(topk), bbox_inches="tight")

      plt.figure()
      for i, k in enumerate(spr_ranks.keys()):
        plt.scatter(times[k][topk], spr_ranks[k][topk], label=k, marker=markers[i], s=150)
      if scores:
        plt.scatter(0, spr_ranks_synflow[topk], label='synflow', marker=markers[i+1], s=80)
      plt.ylabel('Spearman\'s Correlation')
      plt.xlabel('Time (s)')
      plt.title('Topk = %d %%' % topk)
      plt.ylim((0.2,1.1))
      plt.grid(axis='y')
      # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
      plt.savefig('spearman_topk_{}.png'.format(topk), bbox_inches="tight")

    # results = {}
    # results_structured = {}
    # common_ratios = {}
    # spr_ranks = {}

    # for exp_name in args.exp_name:
    #   path_to_results = os.path.join(args.results_dir, exp_name)

    #   if 'stage_2' in exp_name:
    #     fname = 'result_summary_unfreeze_{}.yaml'.format(args.n_unfreeze) #'2' if args.n_unfreeze is None else args.n_unfreeze)
    #     target_ppl = 70 if 'ppl' not in exp_name else int(re.search('ppl_([0-9]+)', exp_name).group(1))
    #     print(target_ppl)
    #     exp_name = 'fear_stage_2'
    #   else:
    #     fname = 'result_summary.yaml'

    #   yaml_file = os.path.join(path_to_results, fname)
    #   with open(yaml_file, 'r') as f:
    #     results[exp_name] = collections.OrderedDict(yaml.safe_load(f))

    # for k, v in results['fear_baseline'].items():
    #   max_step = k.split('_')[-1]
    #   config_name = re.search('(config_[0-9]+)_', k).group(1)
    #   if max_step not in results_structured.keys():
    #     results_structured[max_step] = {}
    #   results_structured[max_step][config_name] = v  
    
    # if 'fear_stage_2' in results.keys():
    #   common_configs_stage2 = np.intersect1d(list(results['fear_stage_1'].keys()), list(results['fear_stage_2'].keys()))
    #   # fear_stage_1 results:
    #   val_ppl_list_gt_for_fear = []
    #   for k in common_configs_stage2:
    #     val_ppl_list_gt_for_fear.append(results['fear_stage_1'][k]['valid_perplexity'])
    #   sorted_ground_truth_for_fear = np.argsort(val_ppl_list_gt_for_fear)
      
    #   # fear_stage_2 results:
    #   val_ppl_list_stage2 = []
    #   timing_stage2 = []
    #   for k in common_configs_stage2:
    #     val_ppl_list_stage2.append(results['fear_stage_2'][k]['valid_perplexity'])
    #     timing_stage2.append(results['fear_stage_2'][k]['train_elapsed'] + results['fear_stage_1'][k][target_ppl]['time'])
    #   sorted_fear = np.argsort(val_ppl_list_stage2)

    # # fear_baseline results:
    # val_ppl_list_gt = {}

    # val_ppl_list_baseline = {}
    # timing_baseline = {}
    # common_configs = {}

    # for max_step, v in results_structured.items():
    #   val_ppl_list_baseline[max_step] = []
    #   timing_baseline[max_step] = []
    #   val_ppl_list_gt[max_step] = []
      
    #   common_configs[max_step] = np.intersect1d(list(results['fear_stage_1'].keys()), list(results_structured[max_step].keys()))
    #   for k in common_configs[max_step]:
    #     val_ppl_list_baseline[max_step].append(results_structured[max_step][k]['valid_perplexity'])
    #     timing_baseline[max_step].append(results_structured[max_step][k]['train_elapsed'])
      
    #     val_ppl_list_gt[max_step].append(results['fear_stage_1'][k]['valid_perplexity'])

    # # extract common ratio and spearmanrank
    # topk_list = [10,20,30,40,50,100]
    # for topk in topk_list:
    #   if 'fear_stage_2' in results.keys():
    #     common_ratio_fear, spr_rank_fear = get_metrics(topk, sorted_ground_truth=sorted_ground_truth_for_fear, sorted_target=sorted_fear, \
    #                                             val_ppl_list_gt=val_ppl_list_gt_for_fear, val_ppl_list_target=val_ppl_list_stage2, common_configs=common_configs_stage2)
          
    #   common_ratios = []
    #   spr_ranks = []
    #   times = []
      
    #   for max_step in val_ppl_list_baseline.keys():
    #     print('------------ {} total number of configs with steps={}'.format(len(val_ppl_list_gt[max_step]), max_step))
    #     sorted_ground_truth = np.argsort(val_ppl_list_gt[max_step])
    #     sorted_baseline = np.argsort(val_ppl_list_baseline[max_step])

    #     common_ratio, spr_rank = get_metrics(topk, sorted_ground_truth=sorted_ground_truth, sorted_target=sorted_baseline, \
    #                                             val_ppl_list_gt=val_ppl_list_gt[max_step], val_ppl_list_target=val_ppl_list_baseline[max_step], 
    #                                             common_configs=common_configs[max_step])
    #     common_ratios.append(common_ratio)
    #     spr_ranks.append(spr_rank)
        
    #     times.append(np.average(timing_baseline[max_step]))

    #   plt.figure()
    #   plt.scatter(times, common_ratios, label='baseline')
    #   if 'fear_stage_2' in results.keys():
    #     plt.scatter(np.average(timing_stage2), common_ratio_fear, label='fear stage 2')
    #   plt.ylabel('Common ratio')
    #   plt.xlabel('Time (s)')
    #   plt.title('Topk = %d %%' % topk)
    #   plt.grid(axis='y')
    #   plt.legend(loc='lower right')
    #   plt.savefig('common_ratio_topk_{}.png'.format(topk), bbox_inches="tight")

    #   plt.figure()
    #   plt.scatter(times, spr_ranks, label='baseline')
    #   if 'fear_stage_2' in results.keys():
    #     plt.scatter(np.average(timing_stage2), spr_rank_fear, label='fear stage 2')
    #   plt.ylabel('Spearman\'s Correlation')
    #   plt.xlabel('Time (s)')
    #   plt.title('Topk = %d %%' % topk)
    #   plt.ylim(top=1)
    #   plt.grid(axis='y')
    #   plt.legend(loc='lower right')
    #   plt.savefig('spearman_topk_{}.png'.format(topk), bbox_inches="tight")


  elif args.analyze_params: # creates a .yaml file containing config names and corresponding parameter sizes
    for exp_name in args.exp_name:
      path_to_results = os.path.join(args.results_dir, exp_name)
      model_configs = recurse_dir(args, exp_name, path_to_results, filetypes='config.yaml')

      params_adaptive_embedding_list = []
      params_adaptive_softmax_list = []
      params_attention_list = []
      params_ff_list = []
      
      n_all_params = {}
      for config_name, model_config in model_configs.items():
        if isinstance(model_config['n_head'], list) and len(model_config['n_head'])>1:
          model = MemTransformerLM_flex(**model_config)
        else:
          model = MemTransformerLM(**model_config)
        model = model.to(device='cpu')
                
        curr_n_all_param, params_adaptive_embedding, params_adaptive_softmax, params_attention, params_ff = process_parameters(model)

        n_all_params[config_name] = {'AdaEmb': float(params_adaptive_embedding), 'Sftmax': float(params_adaptive_softmax), \
                          'Attn': float(params_attention), 'FFN': float(params_ff), 'total': float(curr_n_all_param)}
        print(config_name, n_all_params[config_name])    

        params_adaptive_embedding_list.append(params_adaptive_embedding*100./curr_n_all_param)
        params_adaptive_softmax_list.append(params_adaptive_softmax*100./curr_n_all_param)
        params_attention_list.append(params_attention*100./curr_n_all_param)
        params_ff_list.append(params_ff*100./curr_n_all_param)
      
      print('summarized %d configurations' % len(n_all_params.keys()))
      yaml_file = os.path.join(path_to_results, 'params_summary.yaml')
      with open(yaml_file, 'w') as f:
          yaml.dump(n_all_params, f)

      # create a box plot of parameter size variation across architectures
      fig, ax = plt.subplots()
      data = [params_adaptive_embedding_list, params_adaptive_softmax_list, params_attention_list, params_ff_list]
      bp = ax.boxplot(data, sym='k+', showmeans=True)
      m = [np.mean(d, axis=0) for d in data]
      for i, line in enumerate(bp['medians']):
          x, y = line.get_xydata()[1]
          text = ' μ={:.2f}'.format(m[i])
          if i>0:
            ax.annotate(text, xy=(x-0.2, y+20))
          else:
            ax.annotate(text, xy=(x, y))

      ax.grid(axis='y')
      plt.xticks(range(1, 5), ['AdaEmb', 'Sftmax', 'Attn', 'FFN'])
      plt.savefig('parameter_breakdown_{}.png'.format(exp_name), bbox_inches="tight")
  

  elif args.param_ranking: # 3 plots: 1,2) common_ratio and spearman correlation versus topk for ranking based on parameter size 3) ppl versus parameter size pareto curve
    common_ratios = {}
    spr_ranks = {}
    
    common_ratios_total = {}
    spr_ranks_total = {}
    
    n_params = {}
    n_params_total = {}

    sorted_ground_truth = {}
    val_ppl_list_gt = {}
    
    legend_keys = []
    for exp_name in args.exp_name:
      try:
        idx = re.search('(fear_stage_1)', exp_name).span()[-1]
      except:
        idx = re.search('(fear_stage1)', exp_name).span()[-1]
      legend_key = exp_name[idx+1:].split('_')[-1]
      if len(legend_key)==0:
        legend_key = 'homogeneous'
      legend_keys.append(legend_key)
      
      path_to_results = os.path.join(args.results_dir, exp_name)
      yaml_file = os.path.join(path_to_results, 'result_summary.yaml')
      with open(yaml_file, 'r') as f:
        results_gt = collections.OrderedDict(yaml.safe_load(f))

      yaml_file = os.path.join(path_to_results, 'params_summary.yaml')
      with open(yaml_file, 'r') as f:
          n_all_params = yaml.safe_load(f)

      common_configs = np.intersect1d(list(results_gt.keys()), list(n_all_params.keys()))
      print('analyzing {} architectures'.format(len(common_configs)))

      # min = 100
      # for k in common_configs:
      #   if abs(n_all_params[k]['total']-5e7)<0.1*5e7 and results_gt[k]['valid_perplexity']<min:
      #     min = results_gt[k]['valid_perplexity']
      #     print(min, n_all_params[k]['total']/1e6)

      # fear_stage_1 results:
      val_ppl_list_gt[legend_key] = []
      for k in common_configs:
        val_ppl_list_gt[legend_key].append(results_gt[k]['valid_perplexity'])
      sorted_ground_truth[legend_key] = np.argsort(val_ppl_list_gt[legend_key])

      # n_param results:
      n_params[legend_key] = []
      n_params_total[legend_key] = []
      for k in common_configs:
        n_params[legend_key].append(-(n_all_params[k]['FFN'] + n_all_params[k]['Attn']))
        n_params_total[legend_key].append(-n_all_params[k]['total'])
      sorted_nparams = np.argsort(n_params[legend_key])
      sorted_nparams_total = np.argsort(n_params_total[legend_key])

      if exp_name=='fear_stage1_similar_params':
        common_configs = np.append(common_configs, 'config_12')
        print(common_configs)
        n_params[legend_key].append(-2598391)
        n_params_total[legend_key].append(-7519752)
        val_ppl_list_gt[legend_key].append(45.9)
        sorted_ground_truth[legend_key] = np.argsort(val_ppl_list_gt[legend_key])
        sorted_nparams = np.argsort(n_params[legend_key])
        sorted_nparams_total = np.argsort(n_params_total[legend_key])

      # extract common ratio and spearmanrank
      common_ratios[legend_key] = []
      spr_ranks[legend_key] = []
      common_ratios_total[legend_key] = []
      spr_ranks_total[legend_key] = []
     
      topk_list = range(10,101,10)
      for topk in topk_list:
        common_ratio, spr_rank = get_metrics(topk, sorted_ground_truth=sorted_ground_truth[legend_key], sorted_target=sorted_nparams, \
                                              val_ppl_list_gt=val_ppl_list_gt[legend_key], val_ppl_list_target=n_params[legend_key])
        common_ratios[legend_key].append(common_ratio)
        spr_ranks[legend_key].append(spr_rank)

        common_ratio_total, spr_rank_total = get_metrics(topk, sorted_ground_truth=sorted_ground_truth[legend_key], sorted_target=sorted_nparams_total, \
                                              val_ppl_list_gt=val_ppl_list_gt[legend_key], val_ppl_list_target=n_params_total[legend_key])
        common_ratios_total[legend_key].append(common_ratio_total)
        spr_ranks_total[legend_key].append(spr_rank_total)

    if exp_name=='fear_stage1_similar_params':
      plt.figure()
      colors = ['b', 'g', 'r', 'm', 'y', 'c', 'k']
      markers = ['s', 'o', '*', '^']
      for idx in range(len(common_configs)):
        i = np.where(common_configs=='config_'+str(idx))[0][0]
        plt.scatter(-np.asarray(n_params[legend_key][i]), np.asarray(val_ppl_list_gt[legend_key][i]), label='config_'+str(idx), color=colors[idx//4], marker=markers[idx%4])
      plt.ylabel('Validation PPL')
      plt.xlabel('Total nParams')
      plt.title('Pareto Curve')
      plt.grid(axis='y')
      plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
      plt.savefig('pareto_similar_params.png', bbox_inches="tight")

    plt.figure()
    for k in legend_keys:
      plt.scatter(-np.asarray(n_params[k])[sorted_ground_truth[k]], np.asarray(val_ppl_list_gt[k])[sorted_ground_truth[k]], label=k)
    plt.ylabel('Validation PPL')
    plt.xlabel('Total nParams')
    plt.title('Pareto Curve')
    plt.grid(axis='y')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('pareto_params.png', bbox_inches="tight")

    plt.figure()
    for k in legend_keys:
      plt.scatter(-np.asarray(n_params_total[k])[sorted_ground_truth[k]], np.asarray(val_ppl_list_gt[k])[sorted_ground_truth[k]], label=k)
    plt.ylabel('Validation PPL')
    plt.xlabel('Total nParams')
    plt.title('Pareto Curve')
    plt.grid(axis='y')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('pareto_params_total.png', bbox_inches="tight")
    
    plt.figure()
    for k in legend_keys:
      plt.scatter(topk_list, common_ratios[k], label=k)
    plt.ylabel('Common ratio')
    plt.xlabel('Topk (%)')
    plt.xticks(topk_list)
    # plt.title('ranking based on number of parameters')
    plt.grid(axis='y')
    plt.ylim((0.0,1.1))
    plt.legend(loc='lower right')
    plt.savefig('common_ratio_topk_nparams.png', bbox_inches="tight")

    plt.figure()
    for k in legend_keys:
      plt.scatter(topk_list, common_ratios_total[k], label=k)
    plt.ylabel('Common ratio')
    plt.xlabel('Topk (%)')
    plt.xticks(topk_list)
    # plt.title('ranking based on number of parameters')
    plt.grid(axis='y')
    plt.ylim((0.0,1.1))
    plt.legend(loc='lower right')
    plt.savefig('common_ratio_topk_nparams_total.png', bbox_inches="tight")

    plt.figure()
    for k in legend_keys:
      plt.scatter(topk_list, spr_ranks[k], label=k, marker='^')
    plt.ylabel('Spearman\'s Correlation')
    plt.xlabel('Topk (%)')
    plt.xticks(topk_list)
    plt.ylim((0.0,1.1))
    plt.grid(axis='y')
    plt.legend(loc='lower right')
    # plt.title('ranking based on number of parameters')
    plt.savefig('spearman_topk_nparams.png', bbox_inches="tight")

    plt.figure()
    for k in legend_keys:
      plt.scatter(topk_list, spr_ranks_total[k], label=k, marker='^')
    plt.ylabel('Spearman\'s Correlation')
    plt.xlabel('Topk (%)')
    plt.xticks(topk_list)
    plt.ylim((0.0,1.1))
    plt.grid(axis='y')
    plt.legend(loc='upper right')
    # plt.title('ranking based on number of parameters')
    plt.savefig('spearman_topk_nparams_total.png', bbox_inches="tight")

  
  elif args.similar_params: # plots ppl versus parameter size pareto curve for similar params experiment
    exp_name = 'fear_stage_1_heterogeneous'
    path_to_results = os.path.join('/home/t-mojanj/logdir/nv_xformer_xl/prev_jobs/', exp_name)
    yaml_file = os.path.join(path_to_results, 'result_summary.yaml')
    with open(yaml_file, 'r') as f:
      baseline_configs = collections.OrderedDict(yaml.safe_load(f))
    yaml_file = os.path.join(path_to_results, 'params_summary.yaml')
    with open(yaml_file, 'r') as f:
      baseline_params = yaml.safe_load(f)
    model_configs = recurse_dir(args, exp_name, path_to_results, filetypes='config.yaml')
    for k in baseline_configs.keys():
      baseline_configs[k].update(baseline_params[k])
      baseline_configs[k].update(model_configs[k])

    n_params = {}
    n_params_total = {}
    val_ppl_list_gt = {}
    labels = {}
    for exp_name in args.exp_name:
      path_to_results = os.path.join(args.results_dir, exp_name)
      yaml_file = os.path.join(path_to_results, 'result_summary.yaml')
      with open(yaml_file, 'r') as f:
        results_gt = collections.OrderedDict(yaml.safe_load(f))
      yaml_file = os.path.join(path_to_results, 'params_summary.yaml')
      with open(yaml_file, 'r') as f:
          n_all_params = yaml.safe_load(f)
      model_configs = recurse_dir(args, exp_name, path_to_results, filetypes='config.yaml')

      common_configs = np.intersect1d(list(results_gt.keys()), list(n_all_params.keys()))
      common_configs = np.sort(common_configs)
      print('analyzing {} architectures'.format(len(common_configs)))
      
      for k in common_configs:
        config_name = re.search('(config_[0-9]+)_', k).group(1)
        if config_name in val_ppl_list_gt.keys():
          val_ppl_list_gt[config_name].append(results_gt[k]['valid_perplexity'])
          n_params[config_name].append((n_all_params[k]['FFN'] + n_all_params[k]['Attn']))
          n_params_total[config_name].append(n_all_params[k]['total'])

          labels[config_name].append(get_label(baseline_configs[config_name], model_configs[k]))
        else:
          val_ppl_list_gt[config_name] = [results_gt[k]['valid_perplexity']]
          n_params[config_name] = [(n_all_params[k]['FFN'] + n_all_params[k]['Attn'])]
          n_params_total[config_name] = [n_all_params[k]['total']]
          labels[config_name] = [get_label(baseline_configs[config_name], model_configs[k])]
        
        if '151' in k:
          print(k, results_gt[k]['valid_perplexity'], labels[config_name][-1])
      
    colors = ['b', 'g', 'r', 'm', 'y', 'c', 'k']
    markers = {'n_layer':'.', 'd_inner':'^', 'd_model':'s', 'd_model_d_inner': 'd', 'n_layer_d_model': 'P'}

    valid_labels = ['n_layer', 'd_inner', 'd_model', 'd_model_d_inner']
    for idx, k in enumerate(val_ppl_list_gt.keys()):
      plt.figure()
      plt.scatter(baseline_configs[k]['FFN']+baseline_configs[k]['Attn'], baseline_configs[k]['valid_perplexity'], \
                    color=colors[idx], marker='*', s=250, label='base config')

      for i in range(len(labels[k])):
        if labels[k][i] not in valid_labels:
          continue
        c = 'b' if labels[k][i]=='d_model_d_inner' else colors[idx]
        plt.scatter(n_params[k][i], val_ppl_list_gt[k][i], color=c, \
                    marker=markers[labels[k][i]], s=100, label='scaled '+labels[k][i], zorder=3)
      plt.ylabel('Validation PPL')
      plt.xlabel('Decoder nParams')
      min_x = ((baseline_configs[k]['FFN']+baseline_configs[k]['Attn'])//1e6-1)*1e6
      max_x = (max(n_params[k])//1e6+2)*1e6
      min_y = 5*(np.min(val_ppl_list_gt[k])//5-1).tolist()
      max_y = 5*(baseline_configs[k]['valid_perplexity']//5+1) #5*(max(baseline_configs[k]['valid_perplexity'], np.max(val_ppl_list_gt[k]))//5+1)
      plt.xlim(min_x, max_x)
      plt.ylim(min_y, max_y)
      # plt.title('Pareto Curve')
      plt.grid(axis='y')

      # for j in range(0, len(labels[k]), 3):
      #   x = np.mean(n_params[k][j:j+3])
      #   plt.vlines(x, ymin=5*(min(val_ppl_list_gt[k])//5)-1, ymax=5*(baseline_configs[k]['valid_perplexity']//5+1), \
      #               colors='darkgray', linestyles='dotted')
      
      plt_handles, plt_labels = plt.gca().get_legend_handles_labels()
      newLabels, newHandles = [], []
      for handle, label in zip(plt_handles, plt_labels):
        if label not in newLabels:
          newLabels.append(label)
          newHandles.append(handle)
      plt.legend(newHandles, newLabels, loc='center left', bbox_to_anchor=(1, 0.5))
      
      plt.savefig('pareto_similar_params_{}.png'.format(idx), bbox_inches="tight")


  elif args.cross_step:
    results = {}

    # get baseline
    fname = 'result_summary.yaml'
    yaml_file = os.path.join(os.path.join(args.results_dir, 'fear_stage_1'), fname)
    assert os.path.exists(yaml_file), 'no result summary for the ground-truth job'
    with open(yaml_file, 'r') as f:
      results['fear_stage_1'] = collections.OrderedDict(yaml.safe_load(f))

    # get other experiments
    for exp_name in args.exp_name:
      path_to_results = os.path.join(args.results_dir, exp_name)
      if 'stage_2' in exp_name:
        fname = 'result_summary_unfreeze_{}.yaml'.format(args.n_unfreeze) #'2' if args.n_unfreeze is None else args.n_unfreeze)
        ppl_threshold = 70 if 'ppl' not in exp_name else re.search('ppl_([0-9]+)', exp_name).group(1)
        exp_name = 'fear_stage_2'
      else:
        fname = 'result_summary.yaml'

      yaml_file = os.path.join(path_to_results, fname)
      with open(yaml_file, 'r') as f:
        results[exp_name] = collections.OrderedDict(yaml.safe_load(f))
    
      common_configs = np.intersect1d(list(results['fear_stage_1'].keys()), list(results['fear_stage_2'].keys()))
      print('analyzing {} architectures'.format(len(common_configs)))
      
      # fear_stage_1 results:
      val_ppl_list_stage1 = []
      for k in common_configs:
        val_ppl_list_stage1.append(results['fear_stage_1'][k]['valid_perplexity'])
      sorted_ground_truth = np.argsort(val_ppl_list_stage1)

      # fear_stage_2 results:
      common_ratios = {}
      spr_ranks = {}

      for step in args.step:
        val_ppl_list_stage2 = []
        for k in common_configs:
          val_ppl_list_stage2.append(results['fear_stage_2'][k][step]['valid_perplexity'])
        sorted_fear = np.argsort(val_ppl_list_stage2)
        
        # extract common ratio and spearmanrank
        key = 'step_{}'.format(step)
        print('--------------- ', key)
        common_ratios[key] = []
        spr_ranks[key] = []

        topk_list = range(10,101,10)
        for topk in topk_list:
          common_ratio, spr_rank = get_metrics(topk, sorted_ground_truth=sorted_ground_truth, sorted_target=sorted_fear, \
                                                val_ppl_list_gt=val_ppl_list_stage1, val_ppl_list_target=val_ppl_list_stage2)
          common_ratios[key].append(common_ratio)
          spr_ranks[key].append(spr_rank)

    plt.figure()
    for k in common_ratios.keys():
      plt.plot(topk_list, common_ratios[k], label=k, marker='.', markersize=10)

    plt.ylabel('Common ratio')
    plt.xlabel('Topk (%)')
    plt.xticks(topk_list)
    plt.legend(loc='lower right')
    plt.title('n_unfreeze:{}, ppl:{}'.format(args.n_unfreeze, ppl_threshold))
    plt.grid(axis='y')
    plt.savefig('common_ratio_topk_steps.png', bbox_inches="tight")

    plt.figure()
    for k in spr_ranks.keys():
      plt.plot(topk_list, spr_ranks[k], label=k, marker='.', markersize=10)
      # for i, label in enumerate(spr_ranks[k]):
      #   plt.text(topk_list[i], spr_ranks[k][i]-0.07, '%.2f'%label)

    plt.ylabel('Spearman\'s Correlation')
    plt.xlabel('Topk (%)')
    plt.xticks(topk_list)
    plt.ylim(top=1)
    plt.legend(loc='lower right')
    plt.grid(axis='y')
    plt.title('n_unfreeze:{}, ppl:{}'.format(args.n_unfreeze, ppl_threshold))
    plt.savefig('spearman_topk_steps.png', bbox_inches="tight")


  elif args.animation:
    # groud-truth ranking
    path_to_results = os.path.join(args.results_dir, 'fear_stage_1')
    yaml_file = os.path.join(path_to_results, 'result_summary.yaml')
    with open(yaml_file, 'r') as f:
      results_gt = collections.OrderedDict(yaml.safe_load(f))

    config_names = np.asarray(list(results_gt.keys()))
    val_ppl_list_gt = []
    for k in config_names:
      val_ppl_list_gt.append(results_gt[k]['valid_perplexity'])
    sorted_gt_idx = np.argsort(val_ppl_list_gt)
    sorted_gt = config_names[sorted_gt_idx]

    output_yaml = os.path.join(path_to_results, 'timing_vs_ppl.yaml')
    if os.path.exists(output_yaml):
      with open(output_yaml, 'r') as f:
        results = yaml.safe_load(f)
    else:
      results = {}
      for job in os.listdir(path_to_results):
        path_to_job = os.path.join(path_to_results, job)
        if not os.path.isdir(path_to_job):
          continue
        
        config_name = get_config_name(job)
        
        for file in os.listdir(path_to_job):
          if 'json' in file:
            results[config_name] = {}
            results[config_name]['train_perplexity'] = []
            results[config_name]['valid_perplexity'] = []
            results[config_name]['train_elapsedtime'] = []
            results[config_name]['val_elapsedtime'] = []
            
            with open(os.path.join(path_to_job, file), 'r') as f:
              lines = f.readlines()
              for l in lines:
                dicts = re.search('DLLL (\{.+?\}\})', l).group(1)
                result = json.loads(dicts)
                if 'valid_perplexity' in result['data'].keys():
                  results[config_name]['valid_perplexity'].append(result['data']['valid_perplexity'])
                  results[config_name]['val_elapsedtime'].append(float(result['elapsedtime']))
                elif 'train_perplexity' in result['data'].keys():
                  if result['data']['train_perplexity'] < 100:
                    results[config_name]['train_perplexity'].append(result['data']['train_perplexity'])
                    results[config_name]['train_elapsedtime'].append(float(result['elapsedtime']))
        print(config_name)
      with open(output_yaml, 'w') as f:
        yaml.dump(results, f)
    
    # ppl_threshold = 50
    # for config_name in results.keys():
    #   idx = min(range(len(results[config_name]['valid_perplexity'])), key=lambda i:abs(int(results[config_name]['valid_perplexity'][i]-ppl_threshold)))
    #   print(config_name, results[config_name]['valid_perplexity'])
    #   print(config_name, results[config_name]['valid_perplexity'][idx], results[config_name]['elapsedtime'][idx])

    #   break
    
    plt.figure()
    for idx in range(0, len(sorted_gt), 10): #range(0, 50, 5):
      config_name = sorted_gt[idx]
      # print('--------------', str(idx))
      # print(results[config_name]['valid_perplexity'])
      # print(results[config_name]['val_elapsedtime'])
      plt.plot(results[config_name]['val_elapsedtime'], results[config_name]['valid_perplexity'], label='rank {}'.format(str(idx+1)))
    plt.ylabel('validation ppl')
    plt.xlabel('time (s)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('rank_vs_time_val.png', bbox_inches="tight")

    plt.figure()
    for idx in range(0, len(sorted_gt), 10): #range(0, 50, 5):
      config_name = sorted_gt[idx]
      # print('--------------', str(idx))
      # print(results[config_name]['train_perplexity'])
      # print(results[config_name]['train_elapsedtime'])
      plt.plot(results[config_name]['train_elapsedtime'][::100], results[config_name]['train_perplexity'][::100], label='rank {}'.format(str(idx+1)))
    plt.ylabel('Training ppl')
    plt.xlabel('time (s)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('rank_vs_time_train.png', bbox_inches="tight")
                

  elif args.export_to_csv:
    config_keys = ['d_model', 'd_embed', 'n_layer', 'n_head', 'd_head', 'd_inner', 'div_val']

    for exp_name in args.exp_name:
      path_to_results = os.path.join(args.results_dir, exp_name)
      model_configs = recurse_dir(args, exp_name, path_to_results, filetypes='config.yaml')
      
      yaml_file = os.path.join(path_to_results, 'result_summary.yaml')
      with open(yaml_file, 'r') as f:
        results = collections.OrderedDict(yaml.safe_load(f))

      all_configs = {c:[] for c in config_keys}
      all_configs['val_ppl'] = []
      for config_name, config in model_configs.items():
        try:
          all_configs['val_ppl'].append(results[config_name]['valid_perplexity'])
          for k in config_keys:
            all_configs[k].append(config[k])
        except:
          pass
        
      df = pd.DataFrame(all_configs)
      df.to_csv('./{}_arch_summary.csv'.format(exp_name))     


  else:
    for exp_name in args.exp_name:
      path_to_results = os.path.join(args.results_dir, exp_name)

      results = recurse_dir(args, exp_name, path_to_results, filetypes=['.json', '.log'])
      
      if 'stage_2' in exp_name:
        fname = 'result_summary_unfreeze_{}.yaml'.format('2' if args.n_unfreeze is None else args.n_unfreeze)
      else:
        fname = 'result_summary.yaml'
      yaml_file = os.path.join(path_to_results, fname)
      with open(yaml_file, 'w') as f:
        yaml.dump(results, f)

      target_ppl = 70 if 'ppl' not in exp_name else re.search('ppl_([0-9]+)', exp_name).group(1)
      all_val_ppls = []  
      all_times = []
      all_steps = []
      for k, v in results.items():
        try:
          all_val_ppls.append(v['valid_perplexity'])
        except:
          continue

        if 'stage_1' in exp_name:
          all_times.append(v[target_ppl]['time'])
          all_steps.append(v[target_ppl]['step'])
      
      n_params_best = []
      for k, v in results.items():
        try:
          if v['valid_perplexity'] == min(all_val_ppls):
            n_params_best.append(v['n_params'])
        except:
          continue

      print('best achieved ppl: {:.2f} with n_params: {}'.format(min(all_val_ppls), n_params_best))
      
      plt.hist(all_val_ppls, bins=50)
      plt.xlabel('validation perplexity')
      plt.ylabel('# archs')
      plt.title(exp_name+'_unfreeze_{}'.format('2' if args.n_unfreeze is None else args.n_unfreeze))
      plt.savefig('valid_ppl_'+exp_name+'_unfreeze_{}.png'.format('2' if args.n_unfreeze is None else args.n_unfreeze), bbox_inches="tight")
        
      if 'stage_1' in exp_name:
        plt.figure()
        plt.scatter(all_times, all_val_ppls)
        plt.ylabel('Final validation perplexity')
        plt.xlabel('Time to reach threshold val ppl (s)')
        plt.savefig('val_ppl_vs_time_'+exp_name+'.png', bbox_inches="tight")

        ratio_good = np.sum(np.asarray(all_times)<3750)*100./len(all_times)
        print('ratio of good architectures:', ratio_good)

if __name__ == "__main__":
    args = get_parser()
    
    main(args)

    # for exp_name in args.exp_name:
    #   path_to_results = os.path.join(args.results_dir, exp_name)
    #   yaml_file = os.path.join(path_to_results, 'result_summary.yaml')
    #   with open(yaml_file, 'r') as f:
    #     results_gt = collections.OrderedDict(yaml.safe_load(f))

    #   yaml_file = os.path.join(path_to_results, 'params_summary.yaml')
    #   with open(yaml_file, 'r') as f:
    #       n_all_params = yaml.safe_load(f)

    #   for k in results_gt.keys():
    #     results_gt[k].update(n_all_params[k])
    #     print(k, results_gt[k])