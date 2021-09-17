import os
import types
import numpy as np

import torch
import torch.utils.benchmark as benchmark

from archai.nlp.nvidia_transformer_xl.mem_transformer import MemTransformerLM, forward_predict_memtransformer, predict, \
                                                            AdaptiveEmbedding, DecoderLayer, RelPartialLearnableDecoderLayer, RelLearnableDecoderLayer, \
                                                            MultiHeadAttn, RelPartialLearnableMultiHeadAttn, RelLearnableMultiHeadAttn, \
                                                            PositionwiseFF, ProjectedAdaptiveLogSoftmax


def get_list_of_layers(module, layerType=None):
        # returns a list of layers (optionally with a certain type) that have trainable parameters
        
        submodules = list(module.children())
        list_of_layers = []
        
        if layerType is not None:
            for lt in layerType:
                if isinstance(module, lt):
                    return module
        else:
            if len(submodules)==0 and len(list(module.parameters()))>0:
                return module
        
        for m in submodules:
                try:
                    list_of_layers.extend(get_list_of_layers(m, layerType))
                except TypeError:
                    list_of_layers.append(get_list_of_layers(m, layerType))
        
        return list_of_layers


def get_parameter_breakdown(model, layerType=None, verbose=True):
    layers = get_list_of_layers(model, layerType)
    if verbose:
        print('found {} layers for parameter computation.'.format(len(layers)))
    all_params = {}
    params_decoder = {}
    idx = 0
    for l in layers:
        l_name = l.__class__.__name__+ '_' + str(idx)
        idx += 1
        if isinstance(l, DecoderLayer):
            decoder_layers = get_list_of_layers(model, layerType=[MultiHeadAttn, PositionwiseFF])
            for sub_l in decoder_layers:
                params_decoder['Decoder_'+str(idx)+'_'+sub_l.__class__.__name__] = sum(p.nelement() for p in sub_l.parameters())
        
        all_params[l_name] = sum([p.nelement() for p in l.parameters()])
        
    # print('Per-layer Parameters:', all_params)
    # print('Decoder Parameters:', params_decoder)

    # p_sum = 0
    # for k, p in all_params.items():
    #     if 'Decoder' in k:
    #         p_sum += p

    return all_params, params_decoder


def process_parameters(model, verbose=True):
  params_adaptive_embedding, _ = get_parameter_breakdown(model, layerType=[AdaptiveEmbedding], verbose=verbose)
  params_adaptive_softmax, _ = get_parameter_breakdown(model, layerType=[ProjectedAdaptiveLogSoftmax], verbose=verbose)
  params_attention, _ = get_parameter_breakdown(model, layerType=[MultiHeadAttn, RelPartialLearnableMultiHeadAttn, RelLearnableMultiHeadAttn], verbose=verbose)
  params_ff, _ = get_parameter_breakdown(model, layerType=[PositionwiseFF], verbose=verbose)

  params_adaptive_embedding = np.sum(list(params_adaptive_embedding.values()))
  params_adaptive_softmax = np.sum(list(params_adaptive_softmax.values()))
  params_attention = np.sum(list(params_attention.values()))
  params_ff = np.sum(list(params_ff.values()))

  n_all_param = np.sum(params_adaptive_embedding) + np.sum(params_adaptive_softmax) + np.sum(params_attention) + np.sum(params_ff)
  n_nonemb_param = np.sum(params_attention) + np.sum(params_ff)
  if verbose:
    print('total parameter size:', n_all_param)
    print('nonemb parameter size:', n_nonemb_param)

  # n_all_param_gt = sum([p.nelement() for p in model.parameters()])
  # assert n_all_param_gt == n_all_param, print(n_all_param_gt, n_all_param)
  n_nonemb_param_gt = sum([p.nelement() for p in model.layers.parameters()])
  assert n_nonemb_param_gt == n_nonemb_param, print(n_nonemb_param_gt, n_nonemb_param)

  return n_all_param, params_adaptive_embedding, params_adaptive_softmax, params_attention, params_ff


def get_model_and_params(model_config, verbose=False):
  if isinstance(model_config['n_head'], list) and len(model_config['n_head'])>1:
    model = MemTransformerLM_flex(**model_config)
  else:
    model = MemTransformerLM(**model_config)
  model = model.to(device='cpu')
          
  curr_n_all_param, params_adaptive_embedding, params_adaptive_softmax, params_attention, params_ff = process_parameters(model, verbose=verbose)

  return curr_n_all_param, params_adaptive_embedding, params_adaptive_softmax, params_attention, params_ff


def get_model(model_config, train=False):
    if isinstance(model_config['n_head'], list) and len(model_config['n_head'])>1:
        model = MemTransformerLM_flex(**model_config)
    else:
        model = MemTransformerLM(**model_config)
    
    if not train:
        model.forward = types.MethodType(forward_predict_memtransformer, model)
        model.crit.forward = types.MethodType(predict, model.crit)
        model = model.to(device='cpu')
        model.eval()

    return model


def get_latency(model, model_config, n_threads=1, repeat=10):
    if n_threads > 1:
        torch.set_num_threads(n_threads)
    model = model.to(device='cpu')
    t0 = benchmark.Timer(stmt='model(data)',
                          setup='',
                          globals={'data': torch.LongTensor(model_config['tgt_len']).random_(0, model_config['n_token']).unsqueeze(-1), 'model':model},
                          num_threads=n_threads,
                          label='Multithreaded model execution')
    info = t0.timeit(repeat)
    info._lazy_init()
    latency = info._mean

    return latency


def recurse_dir(pth, filename='config.yaml', path_to_ref=None):
  content = os.listdir(pth) 
  for c in content:
    curr_pth = os.path.join(pth, c)
    if os.path.isfile(curr_pth) and filename in c:
        path_to_ref = curr_pth
    elif os.path.isdir(curr_pth):
      path_to_ref = recurse_dir(curr_pth, filename, path_to_ref)
  
  return path_to_ref