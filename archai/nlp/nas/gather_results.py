import os
from pickle import TRUE
import numpy as np
import collections
from torch import exp_
import yaml
import collections
import argparse
import json
import re
import pprint
import types
import math
import pandas as pd
from functools import partial
from scipy.stats import spearmanr, kendalltau
import plotly.graph_objects as go
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

from archai.nlp.models.model_loader import load_model_from_config
from archai.nlp.nas.nas_utils.constraints.constraint_pipeline import TorchConstraintPipeline

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


def get_statistics(method, results_gt, scores, nparams_dict, topk_list):
  old_keys = list(scores[method].keys())
  for k in old_keys:
    if '_config' in k:
      idx = re.search('(_config)', k).span()[0]
      new_key = k[:idx]
      scores[method][new_key] = scores[method][k]
      del scores[method][k]

  common_configs = np.intersect1d(list(results_gt.keys()), list(scores[method].keys()))
  print('analyzing {} architectures'.format(len(common_configs)))

  # fear_stage_1 results:
  val_ppl_list_gt = []
  for k in common_configs:
    try:
      val_ppl_list_gt.append(results_gt[k]['valid_ppl'])
    except:
      val_ppl_list_gt.append(results_gt[k]['valid_perplexity'])
  sorted_ground_truth = np.argsort(val_ppl_list_gt)

  # zero-cost score results:
  target_scores = []
  for k in common_configs:
    target_scores.append(-scores[method][k])#*1./param_count)   # the higher the score, the better the architecture (reversely correlated with ppl)
  sorted_target = np.argsort(target_scores)

  fig, ax1 = plt.subplots() 
  ax1.plot(range(len(val_ppl_list_gt)), np.asarray(val_ppl_list_gt)[sorted_ground_truth], label='val_ppl')
  ax2 = ax1.twinx() 
  ax2.plot(range(len(val_ppl_list_gt)), np.asarray(target_scores)[sorted_ground_truth], label='# decoder params')
  plt.legend()
  plt.savefig('validation_ppls.png')

  # parameters
  nparams = {}
  for k in common_configs:
    for param_type in nparams_dict[k].keys():
      try:
        nparams[param_type].append(nparams_dict[k][param_type])
      except:
        nparams[param_type] = [nparams_dict[k][param_type]]
  param_corr = {}
  for param_type, target_params in nparams.items():
    param_corr[param_type], _ = spearmanr((-np.asarray(target_scores)).tolist(), target_params)

  common_ratios = []
  spr_ranks = []
  # extract common ratio and spearmanrank
  for topk in topk_list:
    common_ratio, spr_rank = get_metrics(topk, sorted_ground_truth=sorted_ground_truth, sorted_target=sorted_target, \
                                          val_ppl_list_gt=val_ppl_list_gt, val_ppl_list_target=target_scores)
    common_ratios.append(common_ratio)
    spr_ranks.append(spr_rank)

  return common_ratios, spr_ranks, param_corr


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
  # kendal_tau, _ = kendalltau(topk_val_ppl_list_gt, topk_val_ppl_list_target)
  # print('Kendal tau on top %d %% (%d): %.3f'%(topk, len(topk_val_ppl_list_gt), kendal_tau))

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
  key = type if type is not None else None
  
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
            if key is None:
                for k in final_train_log['data'].keys():
                    if 'perplexity' in k or 'ppl' in k:
                        out_dict[k] = final_train_log['data'][k]
            elif key in final_train_log['data'].keys():
                out_dict[key] = final_train_log['data'][key]
                out_dict['amlt_job'] = amlt_job
            break
        except:
            return None
  
  return out_dict


def get_config_name(job):
    idx =  re.search('(config_[0-9]+)', job).span()[0]
    job = job[idx:]
    config_name = job.split('/')[0]
    return config_name + '_' + job.split('/')[1]
    

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
            elif '.yaml' in ft:
              with open(os.path.join(j_path), 'r') as f:
                config = yaml.safe_load(f)
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
                    if config['dataset']=='wt103':
                      config['n_token'] = 267735
                    elif config['dataset']=='lm1b':
                      config['n_token'] = 793470
                    else:
                      raise NotImplementedError
  
              model_config = config
              # if args.model_type == 'mem_transformer':
              #   cutoffs, tie_projs = [], [False]
              #   if config['adaptive']:
              #     assert config['dataset'] in ['wt103', 'wt2', 'lm1b']
              #     if config['dataset'] in ['wt103', 'wt2']:
              #       cutoffs = [19997, 39997, 199997]
              #       tie_projs += [True] * len(cutoffs)
              #     elif config['dataset'] == 'lm1b':
              #       cutoffs = [59997, 99997, 639997]
              #       tie_projs += [False] * len(cutoffs)
              #     model_config['cutoffs'] = cutoffs
              #     model_config['tie_projs'] = tie_projs
              #     model_config['tie_weight'] = config['tied']
              #     model_config['dtype'] = None

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


def get_parser():
  parser = argparse.ArgumentParser(description='Results Analysis.')
  parser.add_argument('--results_dir', type=str, default='/home/v-mojanj/logdir/nv_xformer_xl/prev_jobs/',
                      help='path where amulet results are downloaded')
  parser.add_argument('--exp_name', type=lambda s: [item for item in s.split(',')], #required=True,
                      help='name of maulet experiment')
  parser.add_argument('--model_type', default='mem_transformer', type=str,
                     choices=['hf_gpt2', 'hf_gpt2_flex', 'hf_transfo_xl', 'mem_transformer'],
                     help='Which model type to use')
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
  parser.add_argument('--analyze_params', action='store_true',
                      help='analyze model parameter size')     
  parser.add_argument('--param_ranking', action='store_true',
                      help='generate metrics w.r.t parameter size, separate experiments')   
  parser.add_argument('--param_ranking_all', action='store_true',
                      help='generate metrics w.r.t parameter size,merge all experiments')       
  parser.add_argument('--param_ranking_bins', action='store_true',
                      help='generate metrics w.r.t parameter bins, separate experiments')                  
  parser.add_argument('--scaling_exps', action='store_true',
                      help='plot ppl versus d_model/n_layer for the scaling experiment') 
  parser.add_argument('--finetuning_exps', action='store_true',
                      help='measure correlation between finetuning and param count')
  parser.add_argument('--zerocost_ranking', action='store_true',
                      help='measure correlation between zero-cost proxies and validation ppl')     

  args = parser.parse_args()
  return args


def main(args):
  if args.read_jsons: # extracts information from the .json training log (see get_info_from_json function above)
    for exp_name in args.exp_name:
      path_to_results = os.path.join(args.results_dir, exp_name)

      results = {}
      results = recurse_dir(args, exp_name, path_to_results, filetypes='.json')
        
      print('found %d configurations'%len(results.keys()))
      fname = 'result_summary.yaml'
      yaml_file = os.path.join(path_to_results, fname)
      with open(yaml_file, 'w') as f:
        yaml.dump(results, f)
        print('saved results summary to', fname)

  elif args.analyze_params: # creates a .yaml file containing config names and corresponding parameter sizes
    params_adaptive_embedding_list = []
    params_attention_list = []
    params_ff_list = []
    for exp_name in args.exp_name:
      path_to_results = os.path.join(args.results_dir, exp_name)
      yaml_file = os.path.join(path_to_results, 'params_summary.yaml')
      if os.path.exists(yaml_file):
        with open(yaml_file, 'r') as f:
          n_all_params = yaml.safe_load(f)

          for v in n_all_params.values():
            curr_n_all_param = v['total']
            params_adaptive_embedding_list.append(v['AdaEmb']*100./curr_n_all_param)
            params_attention_list.append(v['Attn']*100./curr_n_all_param)
            params_ff_list.append(v['FFN']*100./curr_n_all_param)

      else:
        model_configs = recurse_dir(args, exp_name, path_to_results, filetypes='config.yaml')
        
        n_all_params = {}
        for config_name, model_config in model_configs.items():
          if args.model_type == 'mem_transformer':
            model_config['d_embed'] = model_config['d_model']
            model_config['d_head'] = -1
          model = load_model_from_config(args.model_type, model_config)
          try:
            print(model.config.div_val)
          except:
            print(model.div_val)
          n_params = model.get_params()
          curr_n_all_param = sum([p.nelement() for p in model.parameters()]) #n_params['total']
          n_all_params[config_name] = {'AdaEmb': float(n_params['embedding']), \
                            'Attn': float(n_params['attention']), 'FFN': float(n_params['ff']), \
                            'nonembedding': float(n_params['non_embedding']), 'total': float(curr_n_all_param)}
          print(config_name, n_all_params[config_name])    

          params_adaptive_embedding_list.append(n_params['embedding']*100./curr_n_all_param)
          params_attention_list.append(n_params['attention']*100./curr_n_all_param)
          params_ff_list.append(n_params['ff']*100./curr_n_all_param)
        
        print('summarized %d configurations' % len(n_all_params.keys()))
        with open(yaml_file, 'w') as f:
            yaml.dump(n_all_params, f)

    # create a box plot of parameter size variation across architectures
    _, ax = plt.subplots()
    data = [params_adaptive_embedding_list, params_attention_list, params_ff_list]
    bp = ax.boxplot(data, sym='k+', showmeans=True, notch=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    # for i, line in enumerate(bp['medians']):
    #     x, y = line.get_xydata()[1]
        # text = ' μ={:.2f}'.format(m[i])
        # if i>0:
        #   ax.annotate(text, xy=(x-0.2, y+20))
        # else:
        #   ax.annotate(text, xy=(x, y))

    ax.grid(axis='y')
    plt.title(args.model_type)
    plt.xticks(range(1, 4), ['AdaEmb', 'Attn', 'FFN'])
    plt.savefig('parameter_breakdown.png', bbox_inches="tight")
  

  elif args.param_ranking: # 3 plots: 1,2) common_ratio and spearman correlation versus topk for ranking based on parameter size 3) ppl versus parameter size pareto curve
    common_ratios = {}
    spr_ranks = {}
    
    common_ratios_total = {}
    spr_ranks_total = {}
    
    n_params = {}
    n_params_total = {}
    div_vals = {}
    all_configs = {}

    sorted_ground_truth = {}
    val_ppl_list_gt = {}
    
    legend_keys = []
    for exp_name in args.exp_name:
      legend_key = 'heterogeneous'
      legend_keys.append(legend_key)
      
      path_to_results = os.path.join(args.results_dir, exp_name)
      yaml_file = os.path.join(path_to_results, 'result_summary.yaml')
      with open(yaml_file, 'r') as f:
        results_gt = collections.OrderedDict(yaml.safe_load(f))

      yaml_file = os.path.join(path_to_results, 'params_summary.yaml')
      with open(yaml_file, 'r') as f:
          n_all_params = yaml.safe_load(f)
      model_configs = recurse_dir(args, exp_name, path_to_results, filetypes='config.yaml')

      # keys = list(model_configs.keys())
      # for k in keys:
        # n_all_params[k.strip('_nv_xformer_xl')] = n_all_params[k]
        # model_configs[k.strip('_nv_xformer_xl')] = model_configs[k]

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
        if model_configs[k]['n_layer'] > 8:
          continue
        try:
          val_ppl_list_gt[legend_key].append(results_gt[k]['valid_ppl'])
        except:
          val_ppl_list_gt[legend_key].append(results_gt[k]['valid_perplexity'])
      sorted_ground_truth[legend_key] = np.argsort(val_ppl_list_gt[legend_key])

      # n_param results:
      n_params[legend_key] = []
      n_params_total[legend_key] = []
      div_vals[legend_key] = []
      all_configs[legend_key] = []
      keys_of_interest = ['d_model', 'n_layer', 'd_inner', 'd_embed', 'div_val', 'n_head']
      for k in common_configs:
        if model_configs[k]['n_layer'] > 8:
          continue
        try:
          n_params[legend_key].append(-(n_all_params[k]['nonembedding']))
        except:
          n_params[legend_key].append(-(n_all_params[k]['Attn']+n_all_params[k]['FFN']))
        n_params_total[legend_key].append(-n_all_params[k]['total'])

        # div_vals[legend_key].append(model_configs[k+'_nv_xformer_xl']['div_val'])
        div_vals[legend_key].append(model_configs[k]['div_val'])
        all_configs[legend_key].append({k2: model_configs[k][k2] for k2 in keys_of_interest})
      sorted_nparams = np.argsort(n_params[legend_key])
      sorted_nparams_total = np.argsort(n_params_total[legend_key])

      print(sorted_nparams[:10], np.asarray(n_params[legend_key])[sorted_nparams[:10]])
      print(sorted_nparams_total[:10], np.asarray(n_params_total[legend_key])[sorted_nparams_total[:10]])

      # extract common ratio and spearmanrank
      common_ratios[legend_key] = []
      spr_ranks[legend_key] = []
      common_ratios_total[legend_key] = []
      spr_ranks_total[legend_key] = []
     
      topk_list = range(10,101,10)
      for topk in topk_list:
        print('*** decoder params')
        common_ratio, spr_rank = get_metrics(topk, sorted_ground_truth=sorted_ground_truth[legend_key], sorted_target=sorted_nparams, \
                                              val_ppl_list_gt=val_ppl_list_gt[legend_key], val_ppl_list_target=n_params[legend_key])
        common_ratios[legend_key].append(common_ratio)
        spr_ranks[legend_key].append(spr_rank)

        print('*** total params')
        common_ratio_total, spr_rank_total = get_metrics(topk, sorted_ground_truth=sorted_ground_truth[legend_key], sorted_target=sorted_nparams_total, \
                                              val_ppl_list_gt=val_ppl_list_gt[legend_key], val_ppl_list_target=n_params_total[legend_key])
        common_ratios_total[legend_key].append(common_ratio_total)
        spr_ranks_total[legend_key].append(spr_rank_total)

    rank_gts = []
    sorted_gt = sorted_ground_truth[legend_key]#[::-1]
    for idx in sorted_nparams:#[::-1]:
      rank_gt = np.where(sorted_gt==idx)[0][0]
      rank_gts.append(rank_gt)

    fig = go.Figure()
    colors = plt.cm.Blues(np.linspace(0.5, 1, 3))
    for k in legend_keys:
      unique_div_vals = np.unique(div_vals[k]).tolist()
      c = [colors[unique_div_vals.index(np.asarray(div_vals[k])[sorted_ground_truth[k]][i])] for i in range(len(n_params[k]))]
      fig.add_trace(go.Scatter(x=-np.asarray(n_params_total[k])[sorted_ground_truth[k]], 
                             y=np.asarray(val_ppl_list_gt[k])[sorted_ground_truth[k]], 
                             mode='markers',
                             marker_color=c,
                             showlegend=True,
                             name='All visited architectures',
                             hovertemplate=None,
                             text=[repr(config) for config in np.asarray(all_configs[k])[sorted_ground_truth[k]]]))

    html_path = f'pareto_params.html'
    fig.write_html(html_path)
    
    
    plt.figure(figsize=(3.5,3.5))
    plt.scatter(range(1, len(sorted_nparams)+1), np.asarray(rank_gts)+1, s=8)
    plt.plot(range(1, len(sorted_nparams)+1), range(1, len(sorted_nparams)+1), linestyle='dashed', color='black')
    plt.xlabel('Proxy Ranking')
    plt.ylabel('Ground-truth Ranking')
    plt.xlim((-1, 200))
    plt.ylim((-1, 200))
    plt.savefig('params_corr.png', bbox_inches="tight", transparent=True)
    
    colors = plt.cm.Blues(np.linspace(0.5, 1, 3))
    plt.figure()
    for k in legend_keys:
      unique_div_vals = np.unique(div_vals[k]).tolist()
      c = [colors[unique_div_vals.index(np.asarray(div_vals[k])[sorted_ground_truth[k]][i])] for i in range(len(n_params[k]))]
      plt.scatter(-np.asarray(n_params[k])[sorted_ground_truth[k]], np.asarray(val_ppl_list_gt[k])[sorted_ground_truth[k]], 
                  label=k, s=15, c=c)
    plt.ylabel('Validation PPL')
    plt.xlabel('Decoder nParams')
    # plt.title('Pareto Curve')
    plt.grid(axis='y')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('pareto_params.png', bbox_inches="tight", transparent=True)

    plt.figure()
    for k in legend_keys:
      unique_div_vals = np.unique(div_vals[k]).tolist()
      c = [colors[unique_div_vals.index(np.asarray(div_vals[k])[sorted_ground_truth[k]][i])] for i in range(len(n_params_total[k]))]
      plt.scatter(-np.asarray(n_params_total[k])[sorted_ground_truth[k]], np.asarray(val_ppl_list_gt[k])[sorted_ground_truth[k]], 
                label=k, s=15, c=c)
    plt.ylabel('Validation PPL')
    plt.xlabel('Total nParams')
    # plt.title('Pareto Curve')
    plt.grid(axis='y')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
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

    plt.figure(figsize=(7,4.2))
    for k in legend_keys:
      plt.plot(topk_list, common_ratios[k], marker='.', markersize=10, label='CR - DecoderParams', color='tab:blue')
      plt.plot(topk_list, spr_ranks[k], marker='^', markersize=10, label='SRC - DecoderParams', color='midnightblue')
      plt.plot(topk_list, common_ratios_total[k], marker='d', linestyle='--',markersize=10, label='CR - TotalParams', color='tab:blue')
      plt.plot(topk_list, spr_ranks_total[k], marker='s', linestyle='--', markersize=10, label='SRC - TotalParams', color='midnightblue')
    # plt.ylabel('Common ratio')
    plt.xlabel('Topk (%)')
    plt.xticks(topk_list)
    # plt.title('ranking based on number of parameters')
    plt.grid(axis='y')
    plt.ylim((0.0,1.1))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.32), ncol=2)
    plt.savefig('common_ratio_spearman_topk_nparams.png', bbox_inches="tight")
  

  elif args.param_ranking_all:
    common_ratios = []
    spr_ranks = []
    
    common_ratios_total = []
    spr_ranks_total = []
    
    n_params = []
    n_params_total = []
    div_vals = []

    sorted_ground_truth = []
    val_ppl_list_gt = []

    results_gt = {}
    n_all_params = {}
    model_configs = {}
    
    for exp_name in args.exp_name:
      legend_key = exp_name
      
      path_to_results = os.path.join(args.results_dir, exp_name)
      yaml_file = os.path.join(path_to_results, 'result_summary.yaml')
      with open(yaml_file, 'r') as f:
        results_gt_old = collections.OrderedDict(yaml.safe_load(f))
      for k, v in results_gt_old.items():
        results_gt[k+'_'+legend_key] = v

      yaml_file = os.path.join(path_to_results, 'params_summary.yaml')
      with open(yaml_file, 'r') as f:
        n_all_params_old = yaml.safe_load(f)
      configs = recurse_dir(args, exp_name, path_to_results, filetypes='config.yaml')
      for k, v in n_all_params_old.items():
        n_all_params[k+'_'+legend_key] = v
        try:
          model_configs[k+'_'+legend_key] = configs[k]
        except:
          model_configs[k+'_'+legend_key] = configs[k + '_nv_xformer_xl']

    common_configs = np.intersect1d(list(results_gt.keys()), list(n_all_params.keys()))
    print('analyzing {} architectures'.format(len(common_configs)))

    # fear_stage_1 results:
    for k in common_configs:
        try:
            val_ppl_list_gt.append(results_gt[k]['valid_perplexity'])
        except:
            val_ppl_list_gt.append(results_gt[k]['valid_ppl'])
    sorted_ground_truth = np.argsort(val_ppl_list_gt)

    # n_param results:
    for k in common_configs:
      try:
        n_params.append(-(n_all_params[k]['nonembedding']))
      except:
        n_params.append(-(n_all_params[k]['Attn']+n_all_params[k]['FFN']))
      n_params_total.append(-n_all_params[k]['total'])
      div_vals.append(model_configs[k]['div_val'])
    sorted_nparams = np.argsort(n_params)
    sorted_nparams_total = np.argsort(n_params_total)

    rank_gts = []
    for idx in sorted_nparams:#[::-1]:
      rank_gt = np.where(sorted_ground_truth==idx)[0][0]
      rank_gts.append(rank_gt)

    print(np.sum(np.absolute(np.asarray(rank_gts)-np.arange(0, len(sorted_nparams))))/len(sorted_nparams))

    sorted_nparams = sorted_nparams[:200]
    rank_gts = rank_gts[:200]
    plt.figure(figsize=(3.5,3.5))
    plt.scatter(range(1, len(sorted_nparams)+1), np.asarray(rank_gts)+1, s=8)
    plt.plot(range(1, len(sorted_nparams)+1), range(1, len(sorted_nparams)+1), linestyle='dashed', color='black')
    plt.xlabel('Proxy Ranking')
    plt.ylabel('Ground-truth Ranking')
    # plt.xlim((-1, 101))
    # plt.ylim((-1, 101))
    plt.savefig(f'params_corr.png', bbox_inches="tight")

    # extract common ratio and spearmanrank
    topk_list = range(10,101,10)
    for topk in topk_list:
      common_ratio, spr_rank = get_metrics(topk, sorted_ground_truth=sorted_ground_truth, sorted_target=sorted_nparams, \
                                            val_ppl_list_gt=val_ppl_list_gt, val_ppl_list_target=n_params)
      common_ratios.append(common_ratio)
      spr_ranks.append(spr_rank)

      common_ratio_total, spr_rank_total = get_metrics(topk, sorted_ground_truth=sorted_ground_truth, sorted_target=sorted_nparams_total, \
                                            val_ppl_list_gt=val_ppl_list_gt, val_ppl_list_target=n_params_total)
      common_ratios_total.append(common_ratio_total)
      spr_ranks_total.append(spr_rank_total)

    plt.figure()
    colors = plt.cm.Blues(np.linspace(0.5, 1, 3))
    unique_div_vals = np.unique(div_vals).tolist()
    c = [colors[unique_div_vals.index(np.asarray(div_vals)[sorted_ground_truth][i])] for i in range(len(n_params))]
    plt.scatter(-np.asarray(n_params)[sorted_ground_truth], np.asarray(val_ppl_list_gt)[sorted_ground_truth], c=c)
    plt.ylabel('Validation PPL')
    plt.xlabel('Decoder nParams')
    plt.grid(axis='y')
    plt.savefig('pareto_params.png', bbox_inches="tight")

    plt.figure(figsize=(5,3))
    c = [colors[unique_div_vals.index(np.asarray(div_vals)[sorted_ground_truth][i])] for i in range(len(n_params_total))]
    plt.scatter(-np.asarray(n_params_total)[sorted_ground_truth], np.asarray(val_ppl_list_gt)[sorted_ground_truth], s=25, c=c)
    plt.ylabel('Validation PPL')
    plt.xlabel('Total nParams')
    plt.grid(axis='y')
    plt.savefig('pareto_params_total.png', bbox_inches="tight")


    plt.figure(figsize=(7,4.2))
    plt.plot(topk_list, common_ratios, marker='.', markersize=10, label='Common Ratio')
    plt.plot(topk_list, spr_ranks, marker='^', markersize=10, label=' Spearman Corr.', color='midnightblue')
    plt.plot(topk_list, common_ratios_total, marker='d', markersize=8, label='Common Ratio-total', color='tab:blue', linestyle='--')
    plt.plot(topk_list, spr_ranks_total, marker='s', markersize=5, label=' Spearman Corr.-total', color='midnightblue', linestyle='--')
    plt.xlabel('Topk (%)')
    plt.xticks(topk_list)
    plt.grid(axis='y')
    plt.ylim((0.0,1.1))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('common_ratio_spearman_topk_nparams.png', bbox_inches="tight")

    # plt.figure(figsize=(7,4.2))
    # plt.plot(topk_list, common_ratios_total, marker='.', markersize=10, label='Common Ratio')
    # plt.plot(topk_list, spr_ranks_total, marker='^', markersize=10, label=' Spearman Correlation')
    # plt.xlabel('Topk (%)')
    # plt.xticks(topk_list)
    # plt.grid(axis='y')
    # plt.ylim((0.0,1.1))
    # plt.legend(loc='lower right')
    # plt.savefig('common_ratio_spearman_topk_nparams_total.png', bbox_inches="tight")

  
  elif args.param_ranking_bins: # plots spearman correlation versus different parameter size bins
    spr_ranks = {}
    spr_ranks_total = {}
    
    n_params = {}
    n_params_total = {}

    sorted_ground_truth = {}
    val_ppl_list_gt = {}
    
    legend_keys = []
    for exp_name in args.exp_name:
      if 'fear' in exp_name:
        try:
          idx = re.search('(fear_stage_1)', exp_name).span()[-1]
        except:
          idx = re.search('(fear_stage1)', exp_name).span()[-1]
        legend_key = exp_name[idx+1:].split('_')[-1]
        if len(legend_key)==0:
          legend_key = 'homogeneous'
      else:
        legend_key = 'heterogeneous'
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

      # fear_stage_1 results:
      val_ppl_list_gt[legend_key] = []
      for k in common_configs:
        val_ppl_list_gt[legend_key].append(results_gt[k]['valid_perplexity'])

      # n_param results:
      n_params[legend_key] = []
      n_params_total[legend_key] = []
      for k in common_configs:
        n_params[legend_key].append(-(n_all_params[k]['FFN'] + n_all_params[k]['Attn']))
        n_params_total[legend_key].append(-n_all_params[k]['total'])

      # extract common ratio and spearmanrank
      spr_ranks[legend_key] = []
      spr_ranks_total[legend_key] = []

      sorted_nparams_all = np.sort(n_params[legend_key])
    
      # bin_list = np.arange(0, 1.01, 0.1)
      # for idx in range(len(bin_list)-1):
      #   start_idx = int(bin_list[idx] * len(sorted_nparams_all))
      #   end_idx = min(math.ceil(bin_list[idx+1] * len(sorted_nparams_all)), len(sorted_nparams_all)-1)
      #   end_param = sorted_nparams_all[end_idx]
      #   start_param = sorted_nparams_all[start_idx]
      #   mask = (n_params[legend_key] >= start_param) * (n_params[legend_key] <= end_param)
      #   mask = np.where(mask)[0].astype(np.int32)
      #   print('-----------------------------')
      #   print(bin_list[idx], start_param, end_param, len(mask))
      
      min_param = sorted_nparams_all[0]
      max_param = sorted_nparams_all[-1]
      param_range = max_param - min_param
      bin_count = 4
      model_sizes = []
      for idx in range(bin_count):
        end_param = (idx+1)*param_range/bin_count + min_param
        start_param = idx*param_range/bin_count + min_param
        mask = (n_params[legend_key] >= start_param) * (n_params[legend_key] < end_param)
        mask = np.where(mask)[0].astype(np.int32)
        model_sizes.append(str(int(-end_param/1e6))+'-'+str(int(-start_param/1e6)))
        print('-----------------------------')
        print(start_param, end_param, len(mask))

        sorted_nparams = np.argsort(np.asarray(n_params[legend_key])[mask])
        sorted_nparams_total = np.argsort(np.asarray(n_params_total[legend_key])[mask])
        sorted_gt = np.argsort(np.asarray(val_ppl_list_gt[legend_key])[mask])
        
        _, spr_rank = get_metrics(100, sorted_ground_truth=sorted_gt, sorted_target=sorted_nparams, \
                                              val_ppl_list_gt=np.asarray(val_ppl_list_gt[legend_key])[mask], \
                                              val_ppl_list_target=np.asarray(n_params[legend_key])[mask])
        spr_ranks[legend_key].append(spr_rank)

        _, spr_rank_total = get_metrics(100, sorted_ground_truth=sorted_gt, sorted_target=sorted_nparams_total, \
                                              val_ppl_list_gt=np.asarray(val_ppl_list_gt[legend_key])[mask], \
                                              val_ppl_list_target=np.asarray(n_params_total[legend_key])[mask])
        spr_ranks_total[legend_key].append(spr_rank_total)

    plt.figure(figsize=(6,3.8))
    for k in legend_keys:
      # plt.bar(bin_list[:-1]+np.diff(bin_list, n=1)/2, spr_ranks[k], width=np.diff(bin_list, n=1))
      bars = plt.bar(model_sizes[::-1], spr_ranks[k][::-1], width=0.5)
    for bar in bars:
      yval = bar.get_height()
      plt.text(bar.get_x(), yval + .005, '%.2f'%yval)
    plt.xlabel(r'# Decoder Params ($\times 10^6$)')
    plt.ylabel('Spearman\'s Correlation')
    plt.grid(axis='y')
    plt.ylim((0.0,1.05))
    plt.savefig('spearman_bins_nparams.png', bbox_inches="tight")

  elif args.scaling_exps:
    exp_name = 'scaling_exps_gpt'
    path_to_results = os.path.join('/home/v-mojanj/logdir/nv_xformer_xl/prev_jobs/', exp_name)
    yaml_file = os.path.join(path_to_results, 'result_summary.yaml')
    with open(yaml_file, 'r') as f:
      results_gt = collections.OrderedDict(yaml.safe_load(f))
    pprint.pprint(results_gt)

    hw_measure = True
    yaml_file = os.path.join(path_to_results, 'latencies.yaml')
    if os.path.exists(yaml_file):
      with open(yaml_file, 'r') as f:
        loaded_latencies = yaml.safe_load(f)
      with open(os.path.join(path_to_results, 'memories.yaml'), 'r') as f:
        loaded_memories = yaml.safe_load(f)
      hw_measure = False
    else:
      loaded_latencies = {}
      loaded_memories = {}

    # setup pipeline for measuring latency and memory
    pipeline = TorchConstraintPipeline(device='cpu')
    
    args.log_type = 'n_nonemb_param'
    n_all_params = recurse_dir(args, exp_name, path_to_results, filetypes='.json')
    model_configs = recurse_dir(args, exp_name, path_to_results, filetypes='model_config.yaml')

    common_configs = np.intersect1d(list(results_gt.keys()), list(n_all_params.keys()))
    common_configs = np.sort(common_configs)
    print('analyzing {} architectures'.format(len(common_configs)))
    
    param_keys = [5e6, 10e6, 25e6, 50e6]
    val_ppls = {k:[] for k in param_keys}
    ratio = {k:[] for k in param_keys}
    latencies = {k:[] for k in param_keys}
    memories = {k:[] for k in param_keys}
    config = {k:[] for k in param_keys}

    for k in common_configs.tolist():
      nparams = n_all_params[k]['n_nonemb_param']
      idx = np.argmin(np.absolute(nparams - np.asarray(param_keys)))
      param_key = param_keys[idx]
      print(k, nparams, param_key)

      if results_gt[k]['valid_ppl'] > 1000 or abs(np.mean(val_ppls[param_key])-results_gt[k]['valid_ppl'])/np.mean(val_ppls[param_key]) > 0.15:
        print('***************')
        print('ppl:{}, d_model:{}, n_layer:{})'.format(results_gt[k]['valid_ppl'], model_configs[k]['d_model'], model_configs[k]['n_layer']))
        print('***************')
        continue
      
      val_ppls[param_key].append(results_gt[k]['valid_ppl'])
      ratio[param_key].append(model_configs[k]['d_model']/model_configs[k]['n_layer'])
      config[param_key].append((model_configs[k]['d_model'], model_configs[k]['n_layer']))

      # measure latency and memory
      if not k in loaded_latencies.keys():
        model = load_model_from_config(args.model_type, model_configs[k])
        proxy, total_params, latency, memory = pipeline(model, model_configs[k])
        assert proxy == nparams
        loaded_latencies[k], loaded_memories[k] = latency, memory
        hw_measure = True
      else:
        latency, memory = loaded_latencies[k], loaded_memories[k]
      latencies[param_key].append(latency)
      memories[param_key].append(memory)

    if hw_measure:
      with open(os.path.join(path_to_results, 'latencies.yaml'), 'w') as f:
        yaml.dump(loaded_latencies, f)
      with open(os.path.join(path_to_results, 'memories.yaml'), 'w') as f:
        yaml.dump(loaded_memories, f)

    fig1, ax1  = plt.subplots(figsize=(7, 5))
    fig2, ax2 = plt.subplots(4, 1, figsize=(14, 10))
    fig3, ax3 = plt.subplots(4, 1, figsize=(14, 10))
    fig_html = go.Figure()
    colors = plt.cm.tab20([2*i for i in range(len(val_ppls.keys()))])
    markers = ['.', '^', 'X', 'd']
    for idx, k in enumerate(val_ppls.keys()):
      indices = np.argsort(ratio[k])
      print(k, (np.max(np.asarray(val_ppls[k])[indices])-np.min(np.asarray(val_ppls[k])[indices]))/np.min(np.asarray(val_ppls[k])[indices]))
      ax1.plot(np.asarray(ratio[k])[indices], np.asarray(val_ppls[k])[indices], marker=markers[idx], color=colors[idx], markersize=10,
                  label='DecoderParams={}M'.format(str(int(k/1e6))))
      indices = np.argsort(latencies[k])
      ax2[idx].plot(np.asarray(latencies[k])[indices]*1000, np.asarray(val_ppls[k])[indices], marker=markers[idx], color=colors[idx], markersize=25,
                  label='DecoderParams={}M'.format(str(int(k/1e6))), linewidth=3)
      indices = np.argsort(memories[k])
      ax3[idx].plot(np.asarray(memories[k])[indices], np.asarray(val_ppls[k])[indices], marker=markers[idx], color=colors[idx], markersize=25,
                  label='DecoderParams={}M'.format(str(int(k/1e6))), linewidth=3)

      print(f'############## ratio range for key {k}', np.min(ratio[k]), np.max(ratio[k]))
      print(f'############## variation compare to mean for key {k}', np.max(abs(np.median(val_ppls[k])-val_ppls[k])/np.median(val_ppls[k])))
      print(f'############## latency range for key {k}', np.max(latencies[k])/np.min(latencies[k]))
      print(f'############## memory range for key {k}', np.max(memories[k])/np.min(memories[k]))
      
      # for i, v in enumerate(val_ppls[k]):
      #   if v > 50:
      #     ax.text(ratio[k][i], v, 'n_layer={}\nd_model={}'.format(config[k][i][1], config[k][i][0]), 
      #               color='black', bbox=dict(facecolor='none', edgecolor='black', pad=2))

      fig_html.add_trace(go.Scatter(x=np.asarray(ratio[k])[indices], 
                                    y=np.asarray(val_ppls[k])[indices], 
                                    mode='lines+markers',
                                    # marker_color=colors[idx],
                                    showlegend=True,
                                    name='DecoderParams={}M'.format(str(int(k/1e6))),
                                    hovertemplate=None,
                                    text=['n_layer={}\nd_model={}'.format(config[k][i][1], config[k][i][0]) for i in indices]))

    ax1.set_ylabel('Validation PPL', fontsize=24)
    fig2.supylabel('Validation PPL', fontsize=48)
    fig3.supylabel('Validation PPL', fontsize=48)
    ranges = [(33, 43), (28, 38), (25, 35), (20, 30)]
    for i, (ax2_, ax3_) in enumerate(zip(ax2, ax3)):
      # ax2_.set_ylabel('Validation PPL', fontsize=24), ax3_.set_ylabel('Validation PPL', fontsize=24)
      # ax2_.set_xlabel('latency (ms)', fontsize=24), ax3_.set_xlabel('Memory (MB)', fontsize=24)
      # ax2_.legend(), ax3_.legend()
      ax2_.set_ylim(ranges[i]), ax3_.set_ylim(ranges[i])
      for xtick in ax2_.xaxis.get_major_ticks():
        xtick.label.set_fontsize(30)
      for ytick in ax2_.yaxis.get_major_ticks():
        ytick.label.set_fontsize(30)
      for xtick in ax3_.xaxis.get_major_ticks():
        xtick.label.set_fontsize(30)
      for ytick in ax3_.yaxis.get_major_ticks():  
        ytick.label.set_fontsize(30)
      ax2_.grid(axis='y'), ax3_.grid(axis='y')
    ax1.set_xlabel(r'$d_{model}/n_{layer}$', fontsize=26)#(r'$\frac{d_{model}}{n_{layer}}$', fontsize=26)
    fig2.supxlabel('Latency (ms)', fontsize=48)
    fig3.supxlabel('Memory (MB)', fontsize=48)
    fig1.tight_layout(), fig2.tight_layout(), fig3.tight_layout()
    ax1.set_xscale("log")
    ax1.set_ylim((20, 50))
    ax1.legend()
    ax1.grid(axis='y')
    fig1.savefig(f'valid_ppl_{exp_name}.png', bbox_inches="tight")
    fig2.savefig(f'valid_ppl_latency_{exp_name}.png', bbox_inches="tight")
    fig3.savefig(f'valid_ppl_memory_{exp_name}.png', bbox_inches="tight")

    fig_html.update_layout(title_text='',
                      xaxis_title=r'$\frac{d_{model}}{n_{layer}}$',
                      yaxis_title='Validation PPL')
    fig_html.update_xaxes(type='log')
    html_path = f'valid_ppl_{exp_name}.html'
    fig_html.write_html(html_path)
  
  elif args.zerocost_ranking:
    path_to_results = os.path.join(args.results_dir, args.exp_name[0])
    # load the ground-truth rankings
    yaml_file = os.path.join(path_to_results, 'result_summary.yaml')
    with open(yaml_file, 'r') as f:
      results_gt = collections.OrderedDict(yaml.safe_load(f))

    with open(os.path.join(path_to_results, 'params_summary.yaml'), 'r') as f:
      nparams_dict = collections.OrderedDict(yaml.safe_load(f))

    costs = {}
    scores = {}
    methods =  ['snip', 'grad_norm', 'fisher', 'jacob_cov', 'grasp', 'jacob_cov_relu', 'synflow']
    for m in methods:
      fname = f'{m}_scores_seed_1111.yaml'
      try:
        with open(os.path.join(path_to_results, fname), 'r') as f:
          scores[m] = yaml.safe_load(f)
      except:
        assert False
        scores[m] = recurse_dir(args, args.exp_name[0], path_to_results, filetypes=fname)
        
      fname = f'{m}_cost.yaml'
      try:
        with open(os.path.join(path_to_results, fname), 'r') as f:
            costs[m] = yaml.safe_load(f)
      except:
        assert False
        costs[m] = recurse_dir(args, args.exp_name[0], path_to_results, filetypes=fname)

      for k in scores[m].keys():
        try:
          scores[m][k] = scores[m][k]['model_config.yaml']
        except:
          continue
        costs[m][k] = costs[m][k]['model_config.yaml']

    costs['nparams'] = {}
    scores['nparams'] = {}
    for k, v in nparams_dict.items():
      try:
        scores['nparams'][k] = v['nonembedding']
      except:
        scores['nparams'][k] = v['FFN'] + v['Attn']
      costs['nparams'][k] = 0.
    
    common_ratios = {}
    spr_ranks = {}
    param_corrs = {}
    topk_list = [10, 30, 50, 100]#range(10,101,10)
    for m in scores.keys():
      print(f'------------ {m} ------------')
      if m=='grasp':
        prev_scores = scores[m]
        scores[m] = {k: -s for k, s in prev_scores.items()}
      common_ratio, spr_rank, param_corr = get_statistics(m, results_gt, scores, nparams_dict, topk_list)
      common_ratios[m] = common_ratio
      spr_ranks[m] = spr_rank
      param_corrs[m] = param_corr
    
    exp_name = args.exp_name[0]
    plt.figure()
    param_types = list(param_corr.keys())
    for i, m in enumerate(common_ratios.keys()):
      corrs = [param_corrs[m][pt] for pt in param_types]
      plt.scatter(range(1, len(param_types)+1), corrs, label=m)#lk+'_method_'+m)
    plt.xticks(range(1, len(param_types)+1), list(param_types))
    plt.legend()
    plt.ylim((0, 1))
    plt.grid(axis='y')
    plt.title('zero-cost score correlation with nparams')
    plt.savefig('zero-cost_params.png', bbox_inches="tight")

    plt.figure()
    for m in common_ratios.keys():
      plt.scatter(topk_list, common_ratios[m], label=m)#lk+'_method_'+m)
    plt.ylabel('Common ratio')
    plt.xlabel('Topk (%)')
    plt.xticks(topk_list)
    plt.title('ranking based on zero-cost methods')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(axis='y')
    plt.savefig(f'common_ratio_topk_zero-cost_{exp_name}.png', bbox_inches="tight")

    plt.figure()
    for i, m in enumerate(common_ratios.keys()):
      plt.scatter(topk_list, spr_ranks[m], label=m)#lk+'_method_'+m)
    plt.ylabel('Spearman\'s Correlation')
    plt.xlabel('Topk (%)')
    plt.xticks(topk_list)
    plt.ylim(top=1)
    plt.grid(axis='y')
    plt.title('ranking based on zero-cost methods')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'spearman_topk_zero-cost_{exp_name}.png', bbox_inches="tight")

    markers = ['o', '^', 's', 'P', '*', 'X', 'd', 'v']
    colors = plt.cm.Blues(np.linspace(0.5, 1, len(markers)))
    labels = {'grad_norm':'grad_norm', 'snip': 'snip', 'fisher':'fisher', 'jacob_cov':'jacob_cov', 'grasp':'grasp', 
              'jacob_cov_relu':'relu_log_det', 'synflow':'synflow', 'nparams':'DecoderParams'}
    plt.figure(figsize=(6, 4))
    for i, m in enumerate(methods + ['nparams']):
      l = labels[m]
      avg_cost = np.mean(list(costs[m].values()))
      plt.scatter(avg_cost, spr_ranks[m][-1], label=l, marker=markers[i], s=180, c=colors[i])#lk+'_method_'+m)
    plt.ylabel('Spearman\'s Correlation')
    plt.xlabel('FLOPs Cost')
    # plt.ylim((0.8, 1.0))
    plt.grid(axis='y')
    # plt.title('ranking based on zero-cost methods')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'family':'monospace'})
    plt.savefig(f'spearman_cost_zero-cost_{exp_name}.png', bbox_inches="tight")

  elif args.finetuning_exps:
    dataset = 'mnli'
    if dataset=='mrpc':
      fname = 'test_results.json'
    elif dataset=='rte':
      fname = 'validation_results.json'
    elif dataset=='mnli':
      fname = 'validation_matched_results.json'
    else:
      raise NotImplementedError
    def find_logs(path_to_dir, fname):
        results = {}
        for j in os.listdir(path_to_dir):
            j_path = os.path.join(path_to_dir, j)
            if os.path.isdir(j_path):
                results.update(find_logs(j_path, fname))
            else:
                if j==fname:
                    with open(j_path, 'r', encoding='utf-8') as f:
                        if '.json' in fname:
                          logs = json.load(f)
                        elif '.yaml' in fname:
                          logs = yaml.safe_load(f)
                    config_name = get_config_name(j_path)
                    print(config_name, logs)
                    results[config_name] = logs
        return results

    finetuning_results = find_logs(os.path.join(args.results_dir, f'{args.exp_name[0]}_{dataset}'), fname)
    results_gt = find_logs(os.path.join(args.results_dir, f'{args.exp_name[0]}_{dataset}'), fname='summary.yaml')
  
    # yaml_file = os.path.join(os.path.join(args.results_dir, args.exp_name[0]), 'params_summary.yaml')
    # with open(yaml_file, 'r') as f:
    #     n_all_params = yaml.safe_load(f)
    # yaml_file = os.path.join(os.path.join(args.results_dir, args.exp_name[0]), 'result_summary.yaml')
    # with open(yaml_file, 'r') as f:
    #     results_gt = collections.OrderedDict(yaml.safe_load(f))

    common_configs = np.intersect1d(list(finetuning_results.keys()), list(results_gt.keys()))
    print('analyzing {} architectures'.format(len(common_configs)))

    acc_list = []
    n_params = []
    valid_ppls = []
    for k in common_configs:
        acc_list.append(finetuning_results[k]['eval_accuracy']) #['eval_combined_score'])
        n_params.append(results_gt[k]['n_nonemb_param']) #n_params.append(n_all_params[k]['nonembedding'])
        valid_ppls.append(results_gt[k]['valid_ppl'])
    sorted_acc = np.argsort(acc_list)
    sorted_nparams = np.argsort(n_params)
    sorted_valid_ppls = np.argsort(valid_ppls)

    plt.figure()
    plt.scatter(n_params, acc_list, s=15, color='midnightblue')
    # plot fit
    p = np.polyfit(n_params, acc_list, 3)
    x_fit = np.linspace(np.min(n_params), np.max(n_params), 300)
    y_fit = np.polyval(p, x_fit)
    plt.plot(x_fit, y_fit, '--')
    plt.ylabel('Finetuning Acc')
    plt.xlabel('Decoder nParams')
    plt.grid(axis='y')
    plt.savefig('ft_acc_versus_params.png', bbox_inches="tight")

    plt.figure()
    plt.scatter(valid_ppls, acc_list, s=15, color='midnightblue')
    p = np.polyfit(valid_ppls, acc_list, 3)
    x_fit = np.linspace(np.min(valid_ppls), np.max(valid_ppls), 300)
    y_fit = np.polyval(p, x_fit)
    plt.plot(x_fit, y_fit, '--')
    plt.ylabel('Finetuning Acc')
    plt.xlabel('Validation PPL')
    plt.grid(axis='y')
    plt.savefig('ft_acc_versus_val_ppl.png', bbox_inches="tight")
    
    # extract common ratio and spearmanrank
    common_ratios = []
    spr_ranks = []
    topk_list = [100] #range(10,101,10)
    for topk in topk_list:
        print('*** acc vs decoder params')
        common_ratio, spr_rank = get_metrics(topk, sorted_ground_truth=sorted_acc, sorted_target=sorted_nparams, \
                                                val_ppl_list_gt=acc_list, val_ppl_list_target=n_params)
        common_ratios.append(common_ratio)
        spr_ranks.append(spr_rank)

        print('*** acc vs valid ppl')
        common_ratio, spr_rank = get_metrics(topk, sorted_ground_truth=sorted_acc, sorted_target=sorted_valid_ppls, \
                                                val_ppl_list_gt=acc_list, val_ppl_list_target=valid_ppls)

if __name__ == "__main__":
    args = get_parser()
    
    main(args)