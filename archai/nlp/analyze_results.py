from builtins import isinstance
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
import glob
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


def get_metrics(topk, sorted_ground_truth, sorted_target, ground_truth, target, common_configs=None):
  # function to compute common ratio and spearman correlation
  idx = int(topk/100.*len(sorted_ground_truth))
  sorted_ground_truth_binned = sorted_ground_truth[:idx].astype(np.int32)
  sorted_target_binned = sorted_target[:idx].astype(np.int32)

  correct = len(np.intersect1d(sorted_target_binned, sorted_ground_truth_binned))
  total = len(sorted_target_binned)
  common_ratio = correct*1./total
  print('Correctly ranked top %d %% (%d) with %.2f accuracy' % (topk, total, correct*1./total))

  topk_val_ppl_list_gt = [ground_truth[i] for i in range(len(ground_truth)) if i in sorted_ground_truth_binned]
  topk_val_ppl_list_target = [target[i] for i in range(len(target)) if i in sorted_ground_truth_binned]
  spr_rank, _ = spearmanr(topk_val_ppl_list_gt, topk_val_ppl_list_target)
  print('Spearman Correlation on top %d %% (%d): %.3f' % (topk, len(topk_val_ppl_list_gt), spr_rank))
  # kendal_tau, _ = kendalltau(topk_val_ppl_list_gt, topk_val_ppl_list_target)
  # print('Kendal tau on top %d %% (%d): %.3f'%(topk, len(topk_val_ppl_list_gt), kendal_tau))

  return common_ratio, spr_rank
                          

def get_config_name(job):
  try:
    idx =  re.search('(config_[0-9]+)', job).span()[0]
    job = job[idx:]
    config_name = job.split('/')[0]
    return config_name + '_' + job.split('/')[1]
  except: 
    config_name =  re.search('(M[0-9]+)', job).group(1)
    return config_name


def recurse_dir(path_to_dir, filename='model_config.yaml'):
  results = {}
  for j in os.listdir(path_to_dir):
      j_path = os.path.join(path_to_dir, j)
      if os.path.isdir(j_path):
        results.update(recurse_dir(j_path, filename))
      else:
        if j == filename:
          with open(os.path.join(j_path), 'r') as f:
            model_config = yaml.load(f, Loader=yaml.Loader)
          config_name = get_config_name(j_path)
          if config_name in results.keys():
            results[config_name].update(model_config)
          else:
            results[config_name] = model_config
  return results


def get_parser():
  parser = argparse.ArgumentParser(description='Results Analysis.')
  parser.add_argument('--results_dir', type=str, default='./archai/nlp/nas/saved_logs',
                      help='path where experiment logs are stored')
  parser.add_argument('--exp_name', type=str,
                      help='name of maulet experiment')
  args = parser.parse_args()
  return args


def main(args):
  '''
  Generates 3 plots: 
    1) gound-truth versus proxy ranking (Fig. 3)
    2) ppl versus parameter size pareto curve (Fig. 6a)
    3) common_ratio and spearman correlation for ranking based on parameter size (Fig. 6b)
  '''
  path_to_results = os.path.join(args.results_dir, args.exp_name)
  args.output_dir = os.path.join(path_to_results, 'plots')
  os.makedirs(args.output_dir, exist_ok=True)
  
  # read logs
  with open(os.path.join(path_to_results, 'ppl_summary.yaml'), 'r') as f:
    logs_val_ppls = collections.OrderedDict(yaml.safe_load(f))
  with open(os.path.join(path_to_results, 'params_summary.yaml'), 'r') as f:
    logs_n_params = yaml.safe_load(f)
  model_configs = recurse_dir(path_to_results, filename='model_config.yaml')
  
  common_configs = np.intersect1d(list(logs_val_ppls.keys()), list(logs_n_params.keys()))
  print(f'Analyzing {len(common_configs)} architectures')

  val_ppls = []
  n_params_nonemb = []
  n_params_total = []
  div_vals = []
  for k in common_configs:
    if isinstance(logs_val_ppls[k], dict):
      val_ppls.append(logs_val_ppls[k]['valid_ppl'])
    else:
      val_ppls.append(logs_val_ppls[k])
    n_params_nonemb.append(-(logs_n_params[k]['nonembedding']))
    n_params_total.append(-logs_n_params[k]['total'])
    div_vals.append(model_configs[k]['div_val'])
  sorted_nparams_nonemb = np.argsort(n_params_nonemb)
  sorted_nparams_total = np.argsort(n_params_total)
  sorted_val_ppl = np.argsort(val_ppls)
  print('decoder param count range:', np.min(np.absolute(n_params_nonemb)), np.max(np.absolute(n_params_nonemb)))
  print('total param count range:', np.min(np.absolute(n_params_total)), np.max(np.absolute(n_params_total)))
  
  # extract common ratio and spearmanrank
  common_ratios_nonemb = []
  spr_ranks_nonemb = []
  common_ratios_total = []
  spr_ranks_total = []
  topk_list = range(10,101,10)
  for topk in topk_list:
    print('*** decoder params')
    cr, src = get_metrics(topk, sorted_ground_truth=sorted_val_ppl, sorted_target=sorted_nparams_nonemb, \
                                          ground_truth=val_ppls, target=n_params_nonemb)
    common_ratios_nonemb.append(cr)
    spr_ranks_nonemb.append(src)

    print('*** total params')
    cr, src = get_metrics(topk, sorted_ground_truth=sorted_val_ppl, sorted_target=sorted_nparams_total, \
                                          ground_truth=val_ppls, target=n_params_total)
    common_ratios_total.append(cr)
    spr_ranks_total.append(src)

  # generate plot of ground-truth ranking versus proxy-based ranking (figure 3)
  ranks_ground_truth = []
  sorted_ground_truth = sorted_val_ppl
  for idx in sorted_nparams_nonemb:
    rank = np.where(sorted_ground_truth==idx)[0][0]
    ranks_ground_truth.append(rank)

  plt.figure(figsize=(3.5,3.5))
  plt.scatter(range(1, len(sorted_nparams_nonemb)+1), np.asarray(ranks_ground_truth)+1, s=8)
  plt.plot(range(1, len(sorted_nparams_nonemb)+1), range(1, len(sorted_nparams_nonemb)+1), linestyle='dashed', color='black')
  plt.xlabel('Proxy Ranking')
  plt.ylabel('Ground-truth Ranking')
  plt.savefig(os.path.join(args.output_dir, 'params_corr.png'), bbox_inches="tight")

  plt.figure(figsize=(5,3))
  colors = ['tab:blue', 'tab:green', 'tab:orange']
  unique_div_vals = np.unique(div_vals).tolist()
  c = [colors[unique_div_vals.index(np.asarray(div_vals)[sorted_val_ppl][i])] for i in range(len(n_params_total))]
  for subcolor in np.unique(c):
    plt.scatter(-np.asarray(n_params_total)[sorted_val_ppl][np.asarray(c)==subcolor], 
                 np.asarray(val_ppls)[sorted_val_ppl][np.asarray(c)==subcolor], 
                 s=25, c=subcolor, label=r'$k$={}'.format(unique_div_vals[colors.index(subcolor)]))
  plt.ylabel('Validation PPL')
  plt.xlabel('Total nParams')
  plt.grid(axis='y')
  plt.legend()
  plt.savefig(os.path.join(args.output_dir, 'pareto_params_total.png'), bbox_inches="tight")

  plt.figure(figsize=(7,4.2))
  plt.plot(topk_list, common_ratios_nonemb, marker='.', markersize=12, label='CR - DecoderParams', color='tab:blue')
  plt.plot(topk_list, spr_ranks_nonemb, marker='^', markersize=10, label='SRC - DecoderParams', color='midnightblue')
  plt.plot(topk_list, common_ratios_total, marker='.', linestyle='--',markersize=12, label='CR - TotalParams', color='tab:blue')
  plt.plot(topk_list, spr_ranks_total, marker='^', linestyle='--', markersize=10, label='SRC - TotalParams', color='midnightblue')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.grid(axis='y')
  plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.32), ncol=2)
  plt.savefig(os.path.join(args.output_dir, 'commonratio_spearman.png'), bbox_inches="tight")

if __name__ == "__main__":
    args = get_parser()
    
    main(args)