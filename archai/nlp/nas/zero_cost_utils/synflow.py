from builtins import NotImplementedError
import os
from pickle import TRUE
import numpy as np
import collections
import yaml
import json
import collections
import argparse
import re
import types
import functools
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import torch
import torch.nn as nn

from archai.nlp.nvidia_transformer_xl.mem_transformer import MemTransformerLM, MemTransformerLM_flex
from archai.nlp.nvidia_transformer_xl.mem_transformer import PositionwiseFF, MultiHeadAttn, RelMultiHeadAttn, \
                                                            RelPartialLearnableMultiHeadAttn, RelLearnableMultiHeadAttn, DecoderLayer, \
                                                            RelLearnableDecoderLayer, RelPartialLearnableDecoderLayer, AdaptiveEmbedding, ProjectedAdaptiveLogSoftmax
from archai.nlp.nvidia_transformer_xl.utils import get_parameter_breakdown, get_list_of_layers, recurse_dir
from archai.nlp.nvidia_transformer_xl import data_utils
from archai.nlp.nvidia_transformer_xl.train import weights_init
from archai.nlp.nvidia_transformer_xl.flops_profile import get_model_flops

from archai.nlp.nvidia_transformer_xl.gather_results import get_metrics, process_parameters
from synflow_utils import compute_synflow_per_weight


def meta_constructor_mapping(loader, node):
    value = loader.construct_mapping(node)
    return value

def meta_constructor_sequence(loader, node):
    value = loader.construct_sequence(node)
    return value

yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar', meta_constructor_sequence)
yaml.add_constructor(u'tag:yaml.org,2002:python/object/apply:numpy.dtype', meta_constructor_mapping)


def _forward_synflow_memformer(self, dec_inp, mems=None):
  qlen, bsz = dec_inp.size()
  word_emb = self.word_emb(dec_inp)

  mlen = mems[0].size(0) if mems is not None else 0
  klen = mlen + qlen
  if self.same_length:
      all_ones = word_emb.new_ones(qlen, klen)
      mask_len = klen - self.mem_len - 1
      if mask_len > 0:
          mask_shift_len = qlen - mask_len
      else:
          mask_shift_len = qlen
      dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                        + torch.tril(all_ones, -mask_shift_len)).bool()
  else:
      dec_attn_mask = torch.triu(word_emb.new_ones(qlen, klen), diagonal=1+mlen).bool()

  hids = []
  # default
  if self.attn_type == 0:
      pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device,
                              dtype=word_emb.dtype)
      if self.clamp_len > 0:
          pos_seq.clamp_(max=self.clamp_len)
      pos_emb = self.pos_emb(pos_seq)

      core_out = self.drop(word_emb)
      pos_emb = self.drop(pos_emb)

      # mask everything to all one for synflow
      core_out = torch.ones_like(core_out, dtype=core_out.dtype, device=core_out.device)
      pos_emb = torch.ones_like(pos_emb, dtype=pos_emb.dtype, device=pos_emb.device)

      for i, layer in enumerate(self.layers):
          hids.append(core_out.detach())
          mems_i = None if mems is None else mems[i]
          core_out = layer(core_out, pos_emb, self.r_w_bias,
                            self.r_r_bias, dec_attn_mask=dec_attn_mask,
                            mems=mems_i)
  else:
    raise NotImplemented

  core_out = self.drop(core_out)

  new_mems = self._update_mems(hids, mems, qlen, mlen)

  return core_out, new_mems

def _forward_synflow_memformer_flex(self, dec_inp, mems=None):
  qlen, bsz = dec_inp.size()
  word_emb = self.word_emb(dec_inp)

  mlen = mems[0].size(0) if mems is not None else 0
  klen = mlen + qlen
  if self.same_length:
      all_ones = word_emb.new_ones(qlen, klen)
      mask_len = klen - self.mem_len - 1
      if mask_len > 0:
          mask_shift_len = qlen - mask_len
      else:
          mask_shift_len = qlen
      dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                        + torch.tril(all_ones, -mask_shift_len)).bool()
  else:
      dec_attn_mask = torch.triu(word_emb.new_ones(qlen, klen), diagonal=1+mlen).bool()

  hids = []
  # default
  if self.attn_type == 0:
      pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device,
                              dtype=word_emb.dtype)
      if self.clamp_len > 0:
          pos_seq.clamp_(max=self.clamp_len)
      pos_emb = self.pos_emb(pos_seq)

      core_out = self.drop(word_emb)
      pos_emb = self.drop(pos_emb)

      # mask everything to all one for synflow
      core_out = torch.ones_like(core_out, dtype=core_out.dtype, device=core_out.device)
      pos_emb = torch.ones_like(pos_emb, dtype=pos_emb.dtype, device=pos_emb.device)

      for i, layer in enumerate(self.layers):
        hids.append(core_out.detach())
        mems_i = None if mems is None else mems[i]

        # core_out = layer(core_out, pos_emb, self.r_w_bias[i], self.r_r_bias[i], dec_attn_mask=dec_attn_mask, mems=mems_i)
        core_out = layer(core_out, pos_emb, getattr(self, f'r_w_bias_{i}'), getattr(self, f'r_r_bias_{i}'), dec_attn_mask=dec_attn_mask, mems=mems_i)
  else:
    raise NotImplemented

  core_out = self.drop(core_out)

  new_mems = self._update_mems(hids, mems, qlen, mlen)

  return core_out, new_mems

def forward_synflow(self, data, target, mems):
    if mems is None:
        mems = self.init_mems()

    tgt_len = target.size(0)
    hidden, new_mems = self._forward(data, mems=mems)

    # pred_hid = hidden[-tgt_len:]
    # return pred_hid.view(-1, pred_hid.size(-1))

    pred_hid = hidden[-tgt_len:]
    out = self.crit(pred_hid.view(-1, pred_hid.size(-1)))
    out = out.view(tgt_len, -1)
    
    return out

def forward_crit(self, hidden, target=None, keep_order=False):
  '''
      hidden :: [len*bsz x d_proj]
  '''
  if self.n_clusters == 0:
    logit = self._compute_logit(hidden, self.out_layers_weights[0], self.out_layers_biases[0], self.get_out_proj(0))
    return logit
  else:
    # construct weights and biases
    weights, biases = [], []
    for i in range(len(self.cutoffs)):
        if self.div_val == 1:
            l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
            weight_i = self.out_layers_weights[0][l_idx:r_idx]
            bias_i = self.out_layers_biases[0][l_idx:r_idx]
        else:
            weight_i = self.out_layers_weights[i]
            bias_i = self.out_layers_biases[i]

        if i == 0:
            weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
            bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)

        weights.append(weight_i)
        biases.append(bias_i)

    head_weight, head_bias, head_proj = weights[0], biases[0], self.get_out_proj(0)
    head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
    return head_logit

def get_scores(args, exp_name):
  model_config_keys = ['n_token', 'n_layer','n_head','d_model','d_head','d_inner','dropout','dropatt', \
                      'd_embed','div_val','pre_lnorm','tgt_len','ext_len','mem_len', \
                      'same_length','attn_type','clamp_len','sample_softmax']
  
  path_to_results = os.path.join(args.results_dir, exp_name) 
  yaml_file_scores = os.path.join(path_to_results, 'synflow_scores_seed_{}.yaml'.format(args.seed))
  yaml_file_cost = os.path.join(path_to_results, 'synflow_cost.yaml')

  calc_scores = not os.path.exists(yaml_file_scores)
  calc_costs = not os.path.exists(yaml_file_cost)
  
  device = torch.device("cpu")
  
  files = []
  dirlist = [path_to_results]
  while len(dirlist) > 0:
    for (dirpath, dirnames, filenames) in os.walk(dirlist.pop()):
      dirlist.extend([os.path.join(dirpath, d) for d in dirnames])
      files.extend(map(lambda n: os.path.join(*n), zip([dirpath] * len(filenames), filenames)))

  if calc_scores or calc_costs:
    scores = {}
    costs = {}
    count = 1
    for f in set(files):
      if 'config.yaml' in f:
        count += 1
        path_to_config = f
        idx =  re.search('(config_[0-9]+)', path_to_config).span()[0]
        job = path_to_config[idx:]
        config_name = job.split('/')[0]
        config_name += '_' + job.split('/')[1]
        with open(path_to_config, 'r') as f:
          config = yaml.full_load(f)
        
        if config is None:
          j_path = os.path.dirname(path_to_config)
          json_file = os.path.join(j_path, 'train_log.json')
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
      
        model_config = {k: config[k] for k in model_config_keys}
          
        # model_config['div_val'] = 1
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
        # model_config['tie_weight'] = False#config['tied']
        model_config['dtype'] = None

        class config_holder():
          pass
        args_init = config_holder
        args_init.proj_init_std = config['proj_init_std']
        args_init.init_std = config['init_std']
        args_init.init_range = config['init_range']
        args_init.init = config['init']
        args_init.emb_init = config['emb_init']
        args_init.emb_init_range = config['emb_init_range']

        if isinstance(model_config['n_head'], list) and len(model_config['n_head'])>1:
          model = MemTransformerLM_flex(**model_config)
          model._forward = types.MethodType(_forward_synflow_memformer_flex, model)
        else:
          model = MemTransformerLM(**model_config)
          model._forward = types.MethodType(_forward_synflow_memformer, model)
        model.forward = types.MethodType(forward_synflow, model)
        model.crit.forward = types.MethodType(forward_crit, model.crit)
        model.apply(functools.partial(weights_init, args=args_init))
        model = model.to(device)
        # print(model)

        # curr_n_all_param, params_adaptive_embedding, params_adaptive_softmax, params_attention, params_ff = process_parameters(model)
        # nparams[config_name] = {'AdaEmb': float(params_adaptive_embedding), 'Sftmax': float(params_adaptive_softmax), \
        #                         'Attn': float(params_attention), 'FFN': float(params_ff), 'total': float(curr_n_all_param)} 
        
        B = 1
        tgt_len, mem_len, ext_len = model_config['tgt_len'], model_config['mem_len'], model_config['ext_len']
        data_len = tgt_len
        data = torch.ones(data_len*B).to(device, torch.long)
        diter = data_utils.LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)
        if calc_scores:
          for idx, (inp, tgt, seqlen, _) in enumerate(diter):
            grads_abs = compute_synflow_per_weight(model, inp, tgt)
            score = np.sum([torch.sum(g).detach().numpy() for g in grads_abs])    
            break
          scores[config_name] = score.tolist()
          print(count, config_name, scores[config_name])
        
        if calc_costs:
          model.eval()
          with torch.no_grad():
            for _, (inp, tgt, _, _) in enumerate(diter):
              curr_flops = get_model_flops(model, inp, tgt)
              total_flops = np.sum([curr_flops[k] for k in ['Attn', 'FFN', 'Sftmax']]).tolist()
              break
          costs[config_name] = 3 * total_flops
          print(count, config_name, costs[config_name])

  if calc_scores:
    with open(yaml_file_scores, 'w') as f:
        yaml.dump(scores, f)
  if calc_costs:
    with open(yaml_file_cost, 'w') as f:
        yaml.dump(costs, f)
  # with open(os.path.join(path_to_results, 'synflow_params.yaml'), 'w') as f:
  #   yaml.dump(nparams, f)

def get_statistics(seed, results_gt, scores, nparams_dict, topk_list):
  common_configs = np.intersect1d(list(results_gt.keys()), list(scores[seed].keys()))
  print('analyzing {} architectures'.format(len(common_configs)))

  # fear_stage_1 results:
  val_ppl_list_gt = []
  for k in common_configs:
    val_ppl_list_gt.append(results_gt[k]['valid_perplexity'])
  sorted_ground_truth = np.argsort(val_ppl_list_gt)

  # zero-cost score results:
  target_scores = []
  for k in common_configs:
    target_scores.append(-scores[seed][k])#*1./param_count)   # the higher the score, the better the architecture (reversely correlated with ppl)
  sorted_target = np.argsort(target_scores)

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

def plot(args):
  common_ratios = {}
  spr_ranks = {}
  param_corrs = {}
  legend_keys = []
  
  for exp_name in args.exp_name:
    path_to_results = os.path.join(args.results_dir, exp_name)
    legend_key = 'heterogeneous' if 'heterogeneous' in exp_name else 'homogeneous'
    legend_keys.append(legend_key)

    # load the ground-truth rankings
    yaml_file = os.path.join(path_to_results, 'result_summary.yaml')
    with open(yaml_file, 'r') as f:
      results_gt = collections.OrderedDict(yaml.safe_load(f))

    with open(os.path.join(path_to_results, 'params_summary.yaml'), 'r') as f:
      nparams_dict = collections.OrderedDict(yaml.safe_load(f))

    scores = {}
    for file in os.listdir(path_to_results):
      if 'synflow_scores_seed' in file:
        if 'old' in file or 'decoderOnly' in file or 'nclusters' in file:
          continue
        seed = re.search('seed_([0-9]+)', file).group(1)
        with open(os.path.join(path_to_results, file), 'r') as f:
          print('loading scores for seed {} from {}'.format(seed, file))
          scores[seed] = yaml.safe_load(f)
      
    common_ratios[legend_key] = {}
    spr_ranks[legend_key] = {}
    param_corrs[legend_key] = {}
    topk_list = range(10,101,10)
    if args.cross_seed:
      for seed in scores.keys():
        common_ratio, spr_rank, param_corr = get_statistics(seed, results_gt, scores, nparams_dict, topk_list)
        common_ratios[legend_key][seed] = common_ratio
        spr_ranks[legend_key][seed] = spr_rank
        param_corrs[legend_key][seed] = param_corr
    else:
      common_ratio, spr_rank, param_corr = get_statistics(str(args.seed), results_gt, scores, nparams_dict, topk_list)
      common_ratios[legend_key][str(args.seed)] = common_ratio
      spr_ranks[legend_key][str(args.seed)] = spr_rank
      param_corrs[legend_key][str(args.seed)] = param_corr
    
  plt.figure()
  param_types = list(param_corr.keys())
  for lk in legend_keys:
    for seed in common_ratios[lk].keys():
      corrs = [param_corrs[lk][seed][pt] for pt in param_types]
      print(corrs)
      plt.scatter(range(1, len(param_types)+1), corrs, label=lk+'_seed_'+seed)
  plt.xticks(range(1, len(param_types)+1), list(param_types))
  plt.legend()
  plt.ylim((0, 1))
  plt.grid(axis='y')
  plt.title('Synflow score correlation with nparams')
  plt.savefig('synflow_params.png', bbox_inches="tight")

  plt.figure()
  for lk in legend_keys:
    for seed in common_ratios[lk].keys():
      plt.scatter(topk_list, common_ratios[lk][seed], label=lk+'_seed_'+seed)
  plt.ylabel('Common ratio')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.title('ranking based on Synflow')
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.grid(axis='y')
  plt.savefig('common_ratio_topk_synflow.png', bbox_inches="tight")

  plt.figure()
  for lk in legend_keys:
    for seed in common_ratios[lk].keys():
      plt.scatter(topk_list, spr_ranks[lk][seed], label=lk+'_seed_'+seed)
  plt.ylabel('Spearman\'s Correlation')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.ylim(top=1)
  plt.grid(axis='y')
  plt.title('ranking based on Synflow')
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.savefig('spearman_topk_synflow.png', bbox_inches="tight")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Results Analysis.')
  parser.add_argument('--results_dir', type=str, default='/home/v-mojanj/logdir/nv_xformer_xl/prev_jobs/',
                      help='path where amulet results are downloaded')
  parser.add_argument('--exp_name', type=lambda s: [item for item in s.split(',')], #required=True,
                      help='name of maulet experiment')
  parser.add_argument('--seed', type=int, default=1111, help='Random seed')
  parser.add_argument('--plot', action='store_true', help='plot the spearman corr and common ratio')
  parser.add_argument('--cross_seed', action='store_true', help='plot the spearman corr and common ratio for all evaluated seeds')
  args = parser.parse_args()

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  for exp_name in args.exp_name:
    get_scores(args, exp_name)
  
  if args.plot:
    plot(args)