import os
from pickle import TRUE
import numpy as np
import collections
import yaml
import collections
import argparse
import re
import types
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import torch

from archai.nlp.nvidia_transformer_xl.mem_transformer import MemTransformerLM #, MemTransformerLM_flex
from archai.nlp.nvidia_transformer_xl.mem_transformer import PositionwiseFF, MultiHeadAttn, RelMultiHeadAttn, \
                                                            RelPartialLearnableMultiHeadAttn, RelLearnableMultiHeadAttn, DecoderLayer, \
                                                            RelLearnableDecoderLayer, RelPartialLearnableDecoderLayer, AdaptiveEmbedding, ProjectedAdaptiveLogSoftmax
from archai.nlp.nvidia_transformer_xl.utils import get_list_of_layers
from archai.nlp.nvidia_transformer_xl import data_utils
from archai.nlp.nvidia_transformer_xl.data_utils import get_lm_corpus
from archai.nlp.nvidia_transformer_xl.nvidia_utils import exp_utils
from archai.common import utils, common

from archai.nlp.nvidia_transformer_xl.gather_results import get_metrics, get_config_name

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

def recurse_dir(args, path_to_dir, verbose=True):
  results = {}
  for j in os.listdir(path_to_dir):
      j_path = os.path.join(path_to_dir, j)
      if os.path.isdir(j_path):
        results.update(recurse_dir(args, j_path, verbose))
      else:
        logs = None
        if 'config.yaml' in j_path:
          with open(os.path.join(j_path), 'r') as f:
            config = yaml.load(f)
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
          logs = {'config': config, 'model_config': model_config, 'path':path_to_dir}
        
        if logs: 
          config_name = get_config_name(j_path) #get_config_name(os.path.basename(os.path.dirname(j_path)))
          if verbose:
            print(config_name, logs)
          results[config_name] = logs
  
  return results


def forward_predict_memtransformer(self, data, target, mems):
  # if mems is None:
  #     mems = self.init_mems()
  mems = None

  tgt_len = target.size(0)
  hidden, new_mems = self._forward(data, mems=mems)

  pred_hid = hidden[-tgt_len:]
  # return pred_hid.view(-1, pred_hid.size(-1))

  if self.sample_softmax > 0 and self.training:
    raise NotImplemented
    # assert self.tie_weight
    # logit = sample_logits(self.word_emb, self.out_layer.bias, target,
    #                         pred_hid, self.sampler)
    # loss = -F.log_softmax(logit, -1)[:, :, 0]
  else:
    output = self.crit.predict(pred_hid.view(-1, pred_hid.size(-1)))
  
  return (output, new_mems)


def get_in_out_shape(self, input, output):
  self.input_size = torch.tensor(input[0].size())
  self.output_size = torch.tensor(output.size())


def get_layer_flops(l):
  if isinstance(l, AdaptiveEmbedding):
    if len(l.emb_projs) > 0:
      return torch.prod(l.output_size) * l.emb_projs[0].size(-1)
    else:
      return torch.tensor([0])
    
  elif isinstance(l, PositionwiseFF):
    return (torch.prod(l.input_size) + torch.prod(l.output_size)) * l.d_inner

  elif isinstance(l, RelPartialLearnableMultiHeadAttn):
    return l.flops

  elif isinstance(l, ProjectedAdaptiveLogSoftmax):
    return l.flops

  else:
    raise NotImplemented


def get_model_flops(model, inp, tgt):
  layers_with_flops = get_list_of_layers(model, layerType=[AdaptiveEmbedding, PositionwiseFF, MultiHeadAttn, RelMultiHeadAttn, \
                                                            RelPartialLearnableMultiHeadAttn, RelLearnableMultiHeadAttn, ProjectedAdaptiveLogSoftmax])
  
  # register forward hooks to record input and output sizes
  hooks = []
  for l in layers_with_flops:
    h = l.register_forward_hook(get_in_out_shape)
    hooks.append(h)
  
  _, mems = model(inp, tgt, None)
  model(inp, tgt, mems)

  flops = {}
  for l in layers_with_flops:
    f = get_layer_flops(l)
    
    if isinstance(l, AdaptiveEmbedding):
      key = 'AdaEmb'
    elif isinstance(l, PositionwiseFF):
      key = 'FFN'
    elif isinstance(l, RelPartialLearnableMultiHeadAttn):
      key = 'Attn'
    elif isinstance(l, ProjectedAdaptiveLogSoftmax):
      key = 'Sftmax'
    else:
      raise NotImplemented
      
    if key in flops.keys():
      flops[key] += f.item()
    else:
      flops[key] = f.item()
    
  return flops


def get_flops(args, exp_name):
  path_to_results = os.path.join(args.results_dir, exp_name)
  
  yaml_file = os.path.join(path_to_results, 'flops_summary.yaml'.format(args.seed))
  if os.path.exists(yaml_file):
    with open(yaml_file, 'r') as f:
      print('Loading flops summary')
      flops = yaml.safe_load(f)
  
  else:
    flops = {}

    configs = recurse_dir(args, path_to_results)
    train_iter = None
    for config_name, all_config in configs.items():
      config = all_config['config']
      model_config = all_config['model_config']

      if isinstance(model_config['n_head'], list) and len(model_config['n_head'])>1:
        model = MemTransformerLM_flex(**model_config)
      else:
        model = MemTransformerLM(**model_config)
      model = model.to(device='cpu')

      # load data
      if train_iter is None:
        path_to_data = common.default_dataroot()
        path_to_data = utils.full_path(os.path.join(path_to_data,'textpred', exp_utils.dataset_dir_name(config['dataset'])))
        corpus = get_lm_corpus(path_to_data, config['dataset'], config['vocab'], max_size=config['vocab_size'])
        train_iter = corpus.get_iterator('train', 1, config['tgt_len'], device='cpu', ext_len=config['ext_len'])
          
      # B = 1 #batch size 
      # tgt_len, mem_len, ext_len = 192, 192, 0
      # data_len = tgt_len * 10
      # data = torch.LongTensor(data_len*B).random_(0, config['n_token']).to(device)
      # train_iter = data_utils.LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)

      model.eval()
      for idx, (inp, tgt, seqlen, _) in enumerate(train_iter):
        curr_flops = get_model_flops(model, inp, tgt)
        break
      total_flops = np.sum([curr_flops[k] for k in curr_flops.keys()]).tolist()
      
      flops[config_name] = curr_flops
      flops[config_name]['total'] = total_flops
      print(config_name, flops[config_name])

    print('summarized %d configurations' % len(flops.keys()))
    with open(yaml_file, 'w') as f:
        yaml.dump(flops, f)


def plot(args):
  common_ratios = {}
  spr_ranks = {}
  
  common_ratios_total = {}
  spr_ranks_total = {}
  
  n_flops = {}
  n_flops_total = {}

  val_ppl_list_gt = {}
  sorted_ground_truth = {}

  legend_keys = []
  for exp_name in args.exp_name:
    path_to_results = os.path.join(args.results_dir, exp_name)
    legend_key = 'heterogeneous' if 'heterogeneous' in exp_name else 'homogeneous'
    legend_keys.append(legend_key)

    # load the ground-truth rankings
    yaml_file = os.path.join(path_to_results, 'result_summary.yaml')
    with open(yaml_file, 'r') as f:
      results_gt = collections.OrderedDict(yaml.safe_load(f))

    yaml_file = os.path.join(path_to_results, 'flops_summary.yaml'.format(args.seed))
    with open(yaml_file, 'r') as f:
      flops = yaml.safe_load(f)
    
    common_configs = np.intersect1d(list(results_gt.keys()), list(flops.keys()))
    print('analyzing {} architectures'.format(len(common_configs)))

    # fear_stage_1 results:
    val_ppl_list_gt[legend_key] = []
    for k in common_configs:
      val_ppl_list_gt[legend_key].append(results_gt[k]['valid_perplexity'])
    sorted_ground_truth[legend_key] = np.argsort(val_ppl_list_gt[legend_key])

    # zero-cost score results:
    n_flops[legend_key] = []
    n_flops_total[legend_key] = []
    for k in common_configs:
      n_flops[legend_key].append(-(flops[k]['FFN'] + flops[k]['Attn']))   # the higher the score, the better the architecture (reversely correlated with ppl)
      n_flops_total[legend_key].append(-flops[k]['total'])
    sorted_flops = np.argsort(n_flops[legend_key])
    sorted_flops_total = np.argsort(n_flops_total[legend_key])

    # extract common ratio and spearmanrank
    common_ratios[legend_key] = []
    spr_ranks[legend_key] = []
    common_ratios_total[legend_key] = []
    spr_ranks_total[legend_key] = []
    
    topk_list = range(10,101,10)
    for topk in topk_list:
      common_ratio, spr_rank = get_metrics(topk, sorted_ground_truth=sorted_ground_truth[legend_key], sorted_target=sorted_flops, \
                                            val_ppl_list_gt=val_ppl_list_gt[legend_key], val_ppl_list_target=n_flops[legend_key])
      common_ratios[legend_key].append(common_ratio)
      spr_ranks[legend_key].append(spr_rank)

      common_ratio_total, spr_rank_total = get_metrics(topk, sorted_ground_truth=sorted_ground_truth[legend_key], sorted_target=sorted_flops_total, \
                                            val_ppl_list_gt=val_ppl_list_gt[legend_key], val_ppl_list_target=n_flops_total[legend_key])
      common_ratios_total[legend_key].append(common_ratio_total)
      spr_ranks_total[legend_key].append(spr_rank_total)
  
  plt.figure()
  for k in legend_keys:
      # plt.scatter(-np.asarray(n_flops_total[k])[sorted_ground_truth[k]], np.asarray(val_ppl_list_gt[k])[sorted_ground_truth[k]], label=k)
      plt.scatter(-np.asarray(n_flops[k])[sorted_ground_truth[k]], np.asarray(val_ppl_list_gt[k])[sorted_ground_truth[k]], label=k)
  plt.ylabel('Validation PPL')
  plt.xlabel('Decoder FLOPs')
  plt.title('Pareto Curve')
  plt.grid(axis='y')
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.savefig('pareto_flops.png', bbox_inches="tight")

  plt.figure()
  for k in legend_keys:
    plt.scatter(topk_list, common_ratios[k], label=k)
  plt.ylabel('Common ratio')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.title('ranking based on number of flops')
  plt.grid(axis='y')
  plt.ylim((0,1))
  plt.legend(loc='lower right')
  plt.savefig('common_ratio_topk_nflops.png', bbox_inches="tight")

  plt.figure()
  for k in legend_keys:
    plt.scatter(topk_list, common_ratios_total[k], label=k)
  plt.ylabel('Common ratio')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.title('ranking based on number of flops')
  plt.grid(axis='y')
  plt.ylim((0,1))
  plt.legend(loc='lower right')
  plt.savefig('common_ratio_topk_nflops_total.png', bbox_inches="tight")

  plt.figure()
  for k in legend_keys:
    plt.scatter(topk_list, spr_ranks[k], label=k)
  plt.ylabel('Spearman\'s Correlation')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.grid(axis='y')
  plt.ylim((0,1))
  plt.legend(loc='lower right')
  plt.title('ranking based on number of flops')
  plt.savefig('spearman_topk_nflops.png', bbox_inches="tight")

  plt.figure()
  for k in legend_keys:
    plt.scatter(topk_list, spr_ranks_total[k], label=k)
  plt.ylabel('Spearman\'s Correlation')
  plt.xlabel('Topk (%)')
  plt.xticks(topk_list)
  plt.grid(axis='y')
  plt.ylim((0,1))
  plt.legend(loc='lower right')
  plt.title('ranking based on number of flops')
  plt.savefig('spearman_topk_nflops_total.png', bbox_inches="tight")

  # create a box plot of parameter size variation across architectures
  fig, ax = plt.subplots()
  flops_adaptive_embedding_list = [flops[k]['AdaEmb'] for k in common_configs]
  flops_adaptive_softmax_list = [flops[k]['Sftmax'] for k in common_configs]
  flops_attention_list = [flops[k]['Attn'] for k in common_configs]
  flops_ff_list = [flops[k]['FFN'] for k in common_configs]
  flops_total_list = [flops[k]['total'] for k in common_configs]
  data = np.asarray([flops_adaptive_embedding_list, flops_adaptive_softmax_list, flops_attention_list, flops_ff_list])/np.asarray(flops_total_list) * 100.
  bp = ax.boxplot(data.tolist(), sym='k+', showmeans=True)
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
  plt.savefig('flops_breakdown.png', bbox_inches="tight")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Results Analysis.')
  parser.add_argument('--results_dir', type=str, default='/home/t-mojanj/logdir/nv_xformer_xl/prev_jobs/',
                      help='path where amulet results are downloaded')
  parser.add_argument('--exp_name', type=lambda s: [item for item in s.split(',')], #required=True,
                      help='name of maulet experiment')
  parser.add_argument('--seed', type=int, default=1111, help='Random seed')
  parser.add_argument('--plot', action='store_true', help='plot the spearman corr and common ratio')
  args = parser.parse_args()

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  for exp_name in args.exp_name:
    get_flops(args, exp_name)
  
  if args.plot:
    plot(args)