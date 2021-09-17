import os
from pickle import TRUE
import numpy as np
import collections
import yaml
import collections
import argparse
from tqdm import tqdm
import re
import math
import time
import types
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as rt

from archai.common import utils, common

from archai.nlp.nvidia_transformer_xl.nvidia_utils import exp_utils
from archai.nlp.nvidia_transformer_xl.data_utils import get_lm_corpus
#from archai.nlp.nvidia_transformer_xl.flops_profile import forward_predict_memtransformer, recurse_dir
#from archai.nlp.nvidia_transformer_xl.mem_transformer import MemTransformerLM, forward_predict_memtransformer, predict # MemTransformerLM_flex
from archai.nlp.nvidia_transformer_xl.nvidia_utils import distributed as nv_distributed
from archai.nlp.nvidia_transformer_xl.mem_transformer_inference import MemTransformerLM, forward_predict_memtransformer

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

def evaluate(eval_iter, model, args):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    # default mem_len==192, eval_tgt_len==192, tgt_len==192
    if args.mem_len == 0:
        model.reset_length(tgt_len=args.eval_tgt_len,
                           ext_len=args.ext_len + args.tgt_len - args.eval_tgt_len,
                           mem_len=args.mem_len
                           )
    else:
        model.reset_length(tgt_len=args.eval_tgt_len,
                           ext_len=args.ext_len,
                           mem_len=args.mem_len + args.tgt_len - args.eval_tgt_len,
                           )

    # Evaluation
    total_len, total_loss = 0, 0.
    with torch.no_grad():
        mems = None
        for i, (data, target, seq_len, warm) in enumerate(eval_iter):
          loss, mems = model(data, target, mems)
          loss = loss.float().mean()
          if warm:
              # assert (mems is None) or mems.size(1) == model.mem_len
              total_loss += seq_len * loss.item()
              total_len += seq_len

    # Switch back to the training mode
    model.reset_length(tgt_len=args.tgt_len,
                       ext_len=args.ext_len,
                       mem_len=args.mem_len)

    return total_loss / total_len


def map_to_device(model, device):
  model = model.to(device)
  if hasattr(model, 'r_emb'):
    if isinstance(model.r_emb, list):
      for idx, _ in enumerate(model.r_emb):
        model.r_emb[idx] = model.r_emb[idx].to(device)
  if hasattr(model, 'r_w_bias'):
    if isinstance(model.r_w_bias, list):
      for idx, _ in enumerate(model.r_w_bias):
        model.r_w_bias[idx] = model.r_w_bias[idx].to(device)
  if hasattr(model, 'r_r_bias'):
    if isinstance(model.r_r_bias, list):
      for idx, _ in enumerate(model.r_r_bias):
        model.r_r_bias[idx] = model.r_r_bias[idx].to(device)
  if hasattr(model, 'r_bias'):
    if isinstance(model.r_bias, list):
      for idx, _ in enumerate(model.r_bias):
        model.r_bias[idx] = model.r_bias[idx].to(device)

  return model


def save_inference_outputs(eval_iter, model, output_file, is_onnx=False):
    # Turn on evaluation mode which disables dropout.
    if not is_onnx:
      model.eval()

    # Evaluation
    all_outputs = None
    with torch.no_grad():
      with tqdm(total=eval_iter.n_batch) as pbar:
        for i, (data, target, seq_len, warm) in enumerate(eval_iter):
          if not is_onnx:
            out = model(data).detach().cpu().numpy()
          else:
            ort_inputs = {model.get_inputs()[0].name: data.detach().cpu().numpy()}
            out = model.run(None, ort_inputs)[0]

          out = out.reshape(-1)
          if all_outputs is not None:
            all_outputs = np.concatenate((all_outputs, out), axis=0)
          else:
            all_outputs = out
          pbar.update(1)

    print('{} total outputs saved'.format(all_outputs.shape))
    np.savetxt(output_file, all_outputs, delimiter=',')


def get_parser():
  parser = argparse.ArgumentParser(description='Results Analysis.')   
  parser.add_argument('--checkpoint', type=str, default='/home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_params_80M/checkpoint_best.pt', help='checkpoint of the model to be exported')
  parser.add_argument('--output_dir', type=str, default='/home/t-gjawahar/archai/amlt/onnx_export', help='directory to store onnx checkpoints')
  args = parser.parse_args()
  return args

def load_checkpoint(path, cuda):
    dst = f'cuda:{torch.cuda.current_device()}' if cuda else torch.device('cpu')
    checkpoint = torch.load(path, map_location=dst)
    return checkpoint

def triu_onnx(x, diagonal=0):
    assert len(x.shape) == 2
    arange = torch.arange(x.size(0), device = x.device)
    arange2 = torch.arange(x.size(1), device = x.device)
    mask = arange.unsqueeze(-1).expand(-1, x.size(1)) <= (arange2 - diagonal)
    return x.masked_fill(mask==0, 0)

def export_word_model():
    args = get_parser()    
    print(f'Loading checkpoint from {args.checkpoint}')
    checkpoint = load_checkpoint(args.checkpoint, False)
    model_config = checkpoint['model_config']
    
    model = MemTransformerLM(**model_config)
    dst = f'cuda:{torch.cuda.current_device()}'
    model.load_state_dict(checkpoint['model_state'])
    #------------ override forward function
    model.forward = types.MethodType(forward_predict_memtransformer, model)
    #model.crit.forward = types.MethodType(predict, model.crit)
    model = map_to_device(model, device='cpu')
    model.eval()

    B = 1 # batch size
    data_len = model_config['tgt_len']
    data = torch.LongTensor(data_len*B).random_(0, model_config['n_token']).unsqueeze(-1).to('cpu')

    #output = model(data, None, None) # char
    output = model(data) # word
    print('Inference done')

    torch.triu = triu_onnx

    inputs = {'data': data}
    torch.onnx.export(model,
                (inputs,),
                f=os.path.join(args.output_dir, args.checkpoint.split("/")[-2]+".onnx"), # output file
                input_names=['data'], # i
                output_names=['out'], # output 
                dynamic_axes={'data': {0: 'sequence', 1: 'batch'}, 'out': {0: 'sequence', 1: 'batch'}},
                do_constant_folding=True,
                use_external_data_format=False,
                enable_onnx_checker=True,
                opset_version=12)
    print('ONNX export complete')

export_word_model()

'''
if __name__ == "__main__":
    args = get_parser()
    device='cuda'

    selected_config = 'config_158' # for fear_stage_1_heterogeneous #'config_67' #for fear_stage_1 #'config_13' # for fear_stage_1_heterogeneous2 

    for exp_name in args.exp_name:
      print(exp_name)
      
      path_to_results = os.path.join(args.results_dir, exp_name)
      yaml_file = os.path.join(path_to_results, 'result_summary.yaml')
      with open(yaml_file, 'r') as f:
        results_gt = collections.OrderedDict(yaml.safe_load(f))

      yaml_file = os.path.join(path_to_results, 'params_summary.yaml')
      with open(yaml_file, 'r') as f:
          n_all_params = yaml.safe_load(f)

      common_configs = np.intersect1d(list(results_gt.keys()), list(n_all_params.keys()))
      print('analyzing {} architectures'.format(len(common_configs)))

      if selected_config is None:
        min = 100
        for k in common_configs:
          if abs(n_all_params[k]['total']-2e7)<2e6 and results_gt[k]['valid_perplexity']<min:
            min = results_gt[k]['valid_perplexity']
            print('%s -> ppl: %.2f, total params: %.2f, decoder params: %.2f' % (k, min, n_all_params[k]['total']/1e6, (n_all_params[k]['Attn']+n_all_params[k]['FFN'])/1e6))
      
      else:
        k = selected_config
        print('%s -> ppl: %.2f, total params: %.2f, decoder params: %.2f' % (k, results_gt[k]['valid_perplexity'], n_all_params[k]['total']/1e6, (n_all_params[k]['Attn']+n_all_params[k]['FFN'])/1e6))

        configs = recurse_dir(args, path_to_results, verbose=False)
        model_config = configs[k]['model_config']
        if 'heterogeneous2' in exp_name:
          model_config['d_head'] = None
        general_config = configs[k]['config']
        path = configs[k]['path']
        
        #------------ download trained checkpoint
        if not os.path.exists(os.path.join(path, 'checkpoint_best.pt')):
          job_name = path.split('/')[path.split('/').index(exp_name)+1]
          run_command = 'amlt results {} :{} -I "*_best.pt"  -o {} --no-md5'.format(exp_name, job_name, args.results_dir)
          os.system(run_command)

        #------------ create model and load trained weights
        if isinstance(model_config['n_head'], list) and len(model_config['n_head'])>1:
          model = MemTransformerLM_flex(**model_config)
        else:
          model = MemTransformerLM(**model_config)
        dst = f'cuda:{torch.cuda.current_device()}'
        print(f'Loading checkpoint from {path}')
        checkpoint = torch.load(os.path.join(path, 'checkpoint_best.pt'))#, map_location=dst)
        if isinstance(model, MemTransformerLM_flex):
          for key in ['r_w_bias', 'r_r_bias']:   # TODO: only support for attention_type=0 now
            for i in range(model.n_layer):
              if f'{key}_{i}' not in checkpoint['model_state'].keys():
                checkpoint['model_state'][f'{key}_{i}'] = torch.Tensor(model.n_heads[i], model.d_heads[i]).zero_()
        model.load_state_dict(checkpoint['model_state'])
        model = map_to_device(model, device='cuda')

        #------------ load validation data
        pt_data_dir, pt_output_dir = common.pt_dirs()
        datadir = pt_data_dir or common.default_dataroot()
        datadir = utils.full_path(os.path.join(datadir,'textpred', exp_utils.dataset_dir_name(general_config['dataset'])))
        corpus = get_lm_corpus(datadir, general_config['dataset'], general_config['vocab'], max_size=general_config['vocab_size'])
        ntokens = len(corpus.vocab)
        vocab = corpus.vocab
        assert model_config['n_token'] == ntokens, 'please match the number of tokens with the loaded model'

        if model_config['mem_len'] == 0: # default is 192
            eval_mem_len = 0
        else:
            eval_mem_len = model_config['mem_len'] + model_config['tgt_len'] - general_config['eval_tgt_len']
        va_iter = corpus.get_iterator('valid', general_config['eval_batch_size'],
                                      general_config['eval_tgt_len'], device='cuda',
                                      mem_len=eval_mem_len, ext_len=model_config['ext_len'])

        #------------ test the loaded model on validation data
        args.mem_len = model_config['mem_len']
        args.eval_tgt_len = general_config['eval_tgt_len']
        args.ext_len = model_config['ext_len']
        args.tgt_len = model_config['tgt_len']
        print('Measuring validation ppl')
        val_loss = evaluate(va_iter, model, args)
        val_loss = nv_distributed.all_reduce_item(val_loss, op='mean')
        curr_ppl = math.exp(val_loss)
        print('Expected ppl: {}, Evaluated ppl: {}'.format(results_gt[k]['valid_perplexity'], curr_ppl))

        #------------ override forward function
        model.forward = types.MethodType(forward_predict_memtransformer, model)
        model.crit.forward = types.MethodType(predict, model.crit)
        # save_inference_outputs(va_iter, model, output_file=os.path.join('onnx_models', 'ground_truth_outputs.txt'))

        model = map_to_device(model, device='cpu')
        model.eval()
        
        torch.set_num_threads(1)
        latency = []
        for _ in range(500):
          data = torch.LongTensor(8).random_(0, model_config['n_token']).unsqueeze(-1)
          start_time = time.time()
          model(data)
          end_time = time.time()
          latency.append(end_time - start_time)
        print(f'Mean inference latency: {np.mean(latency) * 1e3} ms')
        print(f'P50 inference latency: {np.percentile(latency, 50) * 1e3} ms')
        print(f'P90 inference latency: {np.percentile(latency, 90) * 1e3} ms')
        print(f'P95 inference latency: {np.percentile(latency, 95) * 1e3} ms')
        
        exit()
        #------------ export to onnx
        B = 1 # batch size
        data_len = general_config['tgt_len']
        data = torch.LongTensor(data_len*B).random_(0, model_config['n_token']).unsqueeze(-1).to('cpu')

        output = model(data)
        print('Inference done')

        inputs = {'data': data}
        torch.onnx.export(model,
                 (inputs,),
                 f=os.path.join('onnx_models', f'memformer_{k}.onnx'), # output file
                 input_names=['data'], # i
                 output_names=['out'], # output 
                 dynamic_axes={'data': {0: 'sequence', 1: 'batch'}, 'out': {0: 'sequence', 1: 'batch'}},
                 do_constant_folding=True,
                 use_external_data_format=False,
                 enable_onnx_checker=True,
                 opset_version=12)
        print('ONNX export complete')

        model_file = os.path.join('onnx_models', f'memformer_{k}.onnx')
        options = rt.SessionOptions()
        options.enable_profiling = True
        model = rt.InferenceSession(model_file, options)
       
        save_inference_outputs(va_iter, model, output_file=os.path.join('onnx_models', 'onnx_outputs.txt'), is_onnx=True)

        gt_outputs = np.loadtxt(os.path.join('onnx_models', 'ground_truth_outputs.txt'), delimiter=',')
        onnx_outputs = np.loadtxt(os.path.join('onnx_models', 'onnx_outputs.txt'), delimiter=',')
        print('{} matches in {} total'.format(np.sum(gt_outputs==onnx_outputs), len(gt_outputs)))
'''